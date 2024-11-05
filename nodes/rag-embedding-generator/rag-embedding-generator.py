import sys

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import traceback
import json
from urllib.parse import unquote
from langchain_huggingface import HuggingFaceEmbeddings
import os

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

hf_embeddings = None

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
        # Lazy load
        if hf_embeddings is None:
            hf_embeddings = HuggingFaceEmbeddings(model_name=config['modelNameOrPath'])
        # config reload in runtime
        if 'config' in data:
            data_conf = data['config']
            if 'modelNameOrPath' in data_conf:
                config['modelNameOrPath'] = data_conf['modelNameOrPath']
            hf_embeddings = HuggingFaceEmbeddings(model_name=config['modelNameOrPath'])
        if 'documents' in data:
            try:
                documents = data['documents']
            except BaseException as e:
                if os.getenv('DEBUG', '0') == '1':
                    print(type(data), file=sys.__stderr__, flush=True)
                    print(data, file=sys.__stderr__, flush=True)
                    print("\n", file=sys.__stderr__, flush=True)
                    raise e
                else:
                    print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
            texts = [obj['page_content'] if 'page_content' in obj else '' for obj in documents]
            # documents=[Document(page_content=obj['page_content'] if 'page_content' in obj else '',metadata=obj['metadata'] if 'metadata' in obj else None) for obj in documents]
            embeddings = hf_embeddings.embed_documents(texts)
            for idx, x in enumerate(data['documents']):
                documents[idx]['embeddings'] = embeddings[idx]

            # documents=[{"page_content":obj.page_content if hasattr(obj,'page_content') else "","metadata":obj.metadata if hasattr(obj,'metadata') else {}} for obj in documents]
            content = json.dumps({"state": "success", "documents": documents})
            sys.stdout = old_stdout
            print(content + "\n", flush=True)
            sys.stdout = silent_stdout
        else:
            pass
    except BaseException as e:
        if os.getenv('DEBUG', '0') == '1':
            raise e
        else:
            print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
