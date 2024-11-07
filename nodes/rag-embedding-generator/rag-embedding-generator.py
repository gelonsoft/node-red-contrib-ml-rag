import sys
import mmh3

old_stdout = sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout
import base64
import traceback
import json
from langchain_huggingface import HuggingFaceEmbeddings
import os
import numpy as np
import math
import torch
import nltk
import string
from nltk.corpus import stopwords

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

sparse_embeddings_context={
    'init_done':False
}

def init_sparse_embeddings_context():
    if not sparse_embeddings_context['init_done']:
        sparse_embeddings_context['init_done']=True
        sparse_embeddings_context['lemmatizer'] = nltk.WordNetLemmatizer()
        sparse_embeddings_context['punctuation']=set(string.punctuation)
        sparse_embeddings_context['stop_words']=set(stopwords.words('russian'))
        sparse_embeddings_context['stop_words']= sparse_embeddings_context['stop_words'].union(set(stopwords.words('english')))

def rescore(vector):
    new_vector = {}
    for token, value in vector.items():
        token_id = abs(mmh3.hash(token))
        # Examples:
        # Num 0: Log(1/1 + 1) = 0.6931471805599453
        # Num 1: Log(1/2 + 1) = 0.4054651081081644
        # Num 2: Log(1/3 + 1) = 0.28768207245178085
        new_vector[token_id] = math.log(1.0 + value) ** 0.5  # value
    return new_vector


def add_word_weight(word_weights,word,weight):
    word=nltk.word_tokenize(word)[0]
    if word in sparse_embeddings_context['punctuation']:
        return
    if word in sparse_embeddings_context['stop_words']:
        return
    word=sparse_embeddings_context['lemmatizer'].lemmatize(word)

    if word in word_weights:
        word_weights[word] = word_weights[word] + weight
    else:
        word_weights[word] = weight


def convert_outputs_to_sparse_object(tokens,attentions):
    word_weights = {}
    current_word = ""
    current_weight = 0
    for token, weight in zip(tokens[1:-1], attentions[1:-1]):  # Exclude [CLS] and [SEP]
        if token.startswith("##"):
            current_word += token[2:]
            current_weight += weight
        else:
            if current_word:
                add_word_weight(word_weights,current_word,current_weight)
            current_word = token
            current_weight = weight

    if current_word:
        add_word_weight(word_weights,current_word,current_weight)
    word_weights=rescore(word_weights)
    indices, values = zip(*word_weights.items())
    return {'indices':np.array(indices),'values':np.array(values)}

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
            embeddings=None
            with_dense_embeddings=('with_dense_embeddings' in data) and (data['with_dense_embeddings']==0)
            with_bm42_embeddings=('with_bm42_embeddings' in data) and (data['with_bm42_embeddings']==1)
            if with_dense_embeddings:
                embeddings = hf_embeddings.embed_documents(texts)
                for idx, x in enumerate(data['documents']):
                    documents[idx]['embeddings'] = embeddings[idx]

            if with_bm42_embeddings:
                init_sparse_embeddings_context()
                emb_tokenizer=hf_embeddings.client.tokenizer
                emb_model=hf_embeddings.client._first_module().auto_model
                for idx, text in enumerate(texts):
                    inputs = emb_tokenizer(text=text, return_tensors="pt", truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = emb_model(**inputs, output_attentions=True)
                    attentions = np.array(outputs.attentions[-1][0, :, 0].mean(dim=0))
                    tokens = emb_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    documents[idx]['sparse_embeddings'] = convert_outputs_to_sparse_object(tokens,attentions)

            # documents=[{"page_content":obj.page_content if hasattr(obj,'page_content') else "","metadata":obj.metadata if hasattr(obj,'metadata') else {}} for obj in documents]
            content = json.dumps({"state": "success", "documents": documents})
            sys.stdout = old_stdout
            print(base64.b64encode(content.encode()).decode('utf-8')+"\t\t\t\n",flush=True)
            sys.stdout = silent_stdout
        else:
            pass
    except BaseException as e:
        if os.getenv('DEBUG', '0') == '1':
            raise e
        else:
            print(traceback.format_exc() + "\n", file=sys.__stderr__, flush=True)
