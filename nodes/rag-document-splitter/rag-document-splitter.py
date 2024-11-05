import sys

old_stdout=sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import base64
import os
import traceback
import json
from urllib.parse import unquote
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

#read configurations
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

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = config['chunkSize'], #2048
	chunk_overlap = config['chunkOverlap']
)

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

		#config reload in runtime
		if 'config' in data:
			data_conf=data['config']
			if 'chunkSize' in data_conf:
				config['chunkSize']=data_conf['chunkSize']
			if 'chunkSize' in data_conf:
				config['chunkOverlap']=data_conf['chunkOverlap']
			text_splitter = RecursiveCharacterTextSplitter(
				chunk_size = config['chunkSize'], #2048
				chunk_overlap = config['chunkOverlap']
			)

		#print(data)
		if 'documents' in data:
			documents=data['documents']
			documents=[Document(page_content=obj['page_content'] if 'page_content' in obj else '',metadata=obj['metadata'] if 'metadata' in obj else {}) for obj in documents]
			documents=text_splitter.split_documents(documents)
			documents=[{"page_content":obj.page_content if hasattr(obj,'page_content') else "","metadata":obj.metadata if hasattr(obj,'metadata') else {}} for obj in documents]
			content=json.dumps({"state":"success","documents":documents})
			sys.stdout = old_stdout
			print(content+"\n",flush=True)
			sys.stdout = silent_stdout
		else:
			pass
	except BaseException as e:
		if os.getenv('DEBUG','0')=='1':
			raise e
		else:
			print(traceback.format_exc()+"\n",file=sys.__stderr__,flush=True)
