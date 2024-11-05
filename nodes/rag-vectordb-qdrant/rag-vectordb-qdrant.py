import sys

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import traceback
import json
from urllib.parse import unquote
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus
from itertools import islice
import numpy as np
import os
from typing import Iterable, Optional, List, Sequence, Generator, Any, Union
from uuid import uuid4
import base64
if os.environ.get('RAG_DISABLE_SSL_VERIFY', "0") == "1":
    print("Disabling ssl verify")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # os.environ['REQUESTS_CA_BUNDLE'] = 'somepath/rootca.crt'

# read configurations
buf=''
while True:
    msg=input()
    buf=buf+msg
    if "\t\t\t" in msg:
        config = json.loads(base64.b64decode(buf))
        buf=""
        break
    else:
        continue


def print_stdout(data: dict):
    sys.stdout = old_stdout
    content=json.dumps(data)
    print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",flush=True)
    sys.stdout = silent_stdout


def create_client(p_config: dict):
    try:
        if ('url' in p_config) and len(p_config['url']) > 0:
            if ('apiKey' in p_config) and len(p_config['apiKey']) > 0:
                return QdrantClient(url=p_config['url'], api_key=p_config['apiKey'])
            else:
                return QdrantClient(url=p_config['url'])
        elif ('localSavePath' in p_config) and len(p_config['localSavePath']) > 0:
            path = p_config['localSavePath']
            l_dir = os.path.dirname(path)
            if not os.path.isdir(l_dir):
                os.makedirs(l_dir, exist_ok=True)
            return QdrantClient(path=path)
    except BaseException as e:
        print(e, file=sys.__stderr__, flush=True)


main_state={"client":create_client(config)}

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]

def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - np.array(simd.cdist(X, Y, metric="cosine"))
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity

def maximal_marginal_relevance(
        query_embedding: np.ndarray,
        embedding_list: list,
        lambda_mult: float = 0.5,
        k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                    lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs

def _build_payloads(
        texts: Iterable[str],
        metadatas: Optional[List[dict]]
) -> List[dict]:
    payloads = []
    for i, text in enumerate(texts):
        if text is None:
            raise ValueError("At least one of the documents page_content is None. Please remove it before calling")
        metadata = metadatas[i] if metadatas is not None else None
        payloads.append(
            {
                "page_content": text,
                "metadata": metadata,
            }
        )

    return payloads

def _generate_batches(texts: Iterable[str], embeddings: List[float], metadatas: List[dict], ids: Sequence[Any],
                      batch_size: int = 64,
                      ) -> Generator[tuple[list[Any], list[PointStruct]], Any, None]:
    texts_iterator = iter(texts)
    metadatas_iterator = iter(metadatas or [])
    ids_iterator = iter(ids or [uuid4().hex for _ in iter(texts)])

    while batch_texts := list(islice(texts_iterator, batch_size)):
        batch_metadatas = list(islice(metadatas_iterator, batch_size)) or None
        batch_ids = list(islice(ids_iterator, batch_size))
        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(
                batch_ids,
                [{"": vector} for vector in embeddings],
                _build_payloads(batch_texts, batch_metadatas),
            )
        ]

        yield batch_ids, points


def add_document(data):
    if 'collection_name' in data:
        collection_name = data['collection_name']
    else:
        collection_name = 'test'
    embeddings_size = 0
    if 'documents' in data and len(data['documents']) > 0:
        documents = data['documents']
        if 'embeddings' not in documents[0]:
            raise Exception("No 'embeddings' attribute in documents")
        embeddings_size = len(documents[0]['embeddings'])
    else:
        return

    if not (main_state['client'].collection_exists(collection_name)):
        main_state['client'].create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embeddings_size, distance=Distance.COSINE),
        )
    else:
        collection_info = main_state['client'].get_collection(collection_name)
        if collection_info.config.params.vectors.size != embeddings_size:
            raise Exception("'embeddings' size in documents (" + str(
                embeddings_size) + ") <> vector size of collection in storage (" + str(
                collection_info.config.params.vectors.size) + ")")
    ids = [obj['id'] if 'id' in obj else uuid4().hex for obj in documents]
    texts = [obj['page_content'] if 'page_content' in obj else '' for obj in documents]
    metadatas = [obj['metadata'] if 'metadata' in obj else {} for obj in documents]
    embeddings = [obj['embeddings'] if 'embeddings' in obj else [] for obj in documents]
    added_ids = []
    for batch_ids, points in _generate_batches(
            texts, embeddings, metadatas, ids, batch_size=64
    ):
        main_state['client'].upsert(
            collection_name=collection_name, points=points
        )
        added_ids.extend(batch_ids)
        print_stdout({"state": "success", "added_ids": added_ids})


def add_collection(data):
    if 'collection_name' in data:
        collection_name = data['collection_name']
    else:
        raise Exception("Not found collection name in 'collection_name' attribute")

    if 'size' in data:
        collection_size = data['size']
    else:
        raise Exception("Not found collection size in 'size' attribute")

    if not main_state['client'].collection_exists(collection_name):
        main_state['client'].create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=collection_size, distance=Distance.COSINE),
        )
    else:
        if 'drop_if_exists' in data and (data['drop_if_exists'] == 1 or data['drop_if_exists'] == "1"):
            main_state['client'].delete_collection(collection_name)
        else:
            raise Exception("Collection already exists")
        print_stdout({"state": "success"})


def delete_collection(data):
    if 'collection_name' in data:
        collection_name = data['collection_name']
    else:
        raise Exception("Not found collection name in 'collection_name' attribute")

    if not main_state['client'].collection_exists(collection_name):
        return
    else:
        main_state['client'].delete_collection(collection_name)
        print_stdout({"state": "success"})


def delete_documents(data):
    if 'ids' in data:
        ids = data['ids']
    else:
        raise Exception("Not found 'ids' attribute")
    if 'collection_name' in data:
        collection_name = data['collection_name']
    else:
        collection_name = 'test'
    result = main_state['client'].delete(
        collection_name=collection_name,
        points_selector=ids,
    )
    print_stdout({"state": "success", "status": 1 if result.status == UpdateStatus.COMPLETED else 0})

def similarity_search(data):
    collection_name=data['collection_name'] if 'collection_name' in data else 'test'
    limit=int(data['k']) if 'k' in data else 10
    with_payload=int(data['with_payload'])!=0 if 'with_payload' in data else True
    with_vectors=int(data['with_vectors'])!=0 if 'with_vectors' in data else False
    search_type=data['search_type'] if 'search_type' in data else 'mmr'
    score_threshold=data['score_threshold'] if 'score_threshold' in data else 0.0
    if 'embeddings' in data:
        embeddings=data['embeddings']
    else:
        raise Exception("No 'embeddings' attribute exists")
    if not main_state['client'].collection_exists(collection_name):
        raise Exception("Collection not exists")
    results=main_state['client'].query_points(
        query=embeddings,
        collection_name=collection_name,
        limit=limit+15 if search_type=='mmr' else limit,
        with_payload=with_payload,
        with_vectors=True if search_type=='mmr' else with_vectors,
        score_threshold=score_threshold,
        using=""
    ).points

    if search_type=='mmr':
        embeddings_new = [
            result.vector
            if isinstance(result.vector, list)
            else result.vector.get("")  # type: ignore
            for result in results
        ]
        mmr_selected = maximal_marginal_relevance(
            np.array(embeddings), embeddings_new, k=limit, lambda_mult=0.5
        )
        documents = [
            {"page_content": results[i].payload.get('page_content') or None
                , "metadata": results[i].payload.get('metadata') or None
                , "embeddings": results[i].vector or None
                , "id": results[i].id
                ,"score": results[i].score
             }
            for i in mmr_selected
        ]
    else:
        documents = [
            {"page_content": obj.payload.get('page_content') or None
                , "metadata": obj.payload.get('metadata') or None
                , "embeddings": obj.vector or None
                , "id": obj.id
                ,"score": obj.score
             }
            for obj in results
        ]
    print_stdout({"state": "success", "documents": documents})

def scroll(data):
    collection_name=data['collection_name'] if 'collection_name' in data else 'test'
    offset=data['offset'] if 'offset' in data else 0
    limit=int(data['limit']) if 'limit' in data else 10
    filter=None
    if 'filter' in data:
        pass
    with_payload=int(data['with_payload'])!=0 if 'with_payload' in data else True
    with_vectors=int(data['with_vectors'])!=1 if 'with_vectors' in data else False
    order_by=None #data['order_by'] if 'order_by' in data else {}

    if not main_state['client'].collection_exists(collection_name):
        raise Exception("Collection not exists")


    results = main_state['client'].scroll(collection_name=collection_name, with_payload=with_payload, with_vectors=with_vectors, offset=offset,limit=limit)
    documents = [
        {"page_content": obj.payload.get('page_content') or None
            ,"metadata": obj.payload.get('metadata') or None
            ,"embeddings": obj.vector or None
            ,"id": obj.id
         }
        for obj in results[0]
    ]
    print_stdout({"state": "success", "documents": documents,"offset":results[1]})


def query_by_ids(data):
    if 'ids' in data:
        ids = data['ids']
    else:
        raise Exception("Not found 'ids' attribute")
    if 'collection_name' in data:
        collection_name = data['collection_name']
    else:
        collection_name = 'test'
    with_payload=int(data['with_payload'])!=0 if 'with_payload' in data else True
    with_vectors=int(data['with_vectors'])!=1 if 'with_vectors' in data else False
    results = main_state['client'].retrieve(collection_name, ids, with_payload=with_payload,with_vectors=with_vectors)
    # documents=[]
    # if 'format' in data and data['format'] == "extended":
    documents = [
        {"page_content": obj.payload.get('page_content') or None
            , "metadata": obj.payload.get('metadata') or None
            , "embeddings": obj.vector or None
            , "id": obj.id
         }
        for obj in results
    ]
    print_stdout({"state": "success", "documents": documents})


def main_cycle(data):
    if 'config' in data:
        data_conf = data['config']
        if 'apiKey' in data_conf:
            config['apiKey'] = data_conf['apiKey']
        if 'url' in data_conf:
            config['url'] = data_conf['url']
        if 'localSavePath' in data_conf:
            config['localSavePath'] = data_conf['localSavePath']
        main_state['client'] = create_client(config)

    if 'command' in data:
        command = data['command']
        if command == 'add_document':
            add_document(data)
        elif command == 'add_collection':
            add_collection(data)
        elif command == 'delete_collection':
            delete_collection(data)
        elif command == 'delete_documents':
            delete_documents(data)
        elif command == 'scroll':
            scroll(data)
        elif command == 'query_by_ids':
            query_by_ids(data)
        elif command == 'similarity_search':
            similarity_search(data)

    else:
        pass

while True:
    msg=input()
    buf=buf+msg
    #read request
    try:
        if "\t\t\t" in msg:
            data = json.loads(base64.b64decode(buf))
            buf=""
        else:
            continue
        # config reload in runtime
        main_cycle(data)
    except BaseException as e:
        if os.getenv('DEBUG','0')=='1':
            raise e
        else:
            print(traceback.format_exc()+"\n",file=sys.__stderr__,flush=True)
