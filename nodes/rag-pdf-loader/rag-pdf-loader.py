from __future__ import annotations
import sys

old_stdout=sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import json
from urllib.parse import unquote, quote
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.document_loaders.base import BaseLoader
import typing
from langchain_core.documents.base import Blob, Document
import base64

class CustomPDFLoader(BaseLoader):
	def __init__(self, pBytes: bytes, file_path: str='', password: typing.Optional[typing.Union[str, bytes]] = None,
				 extract_images: bool = False):
		self.pBytes = pBytes
		self.file_path=file_path
		self.parser = PyPDFParser(password=password, extract_images=extract_images)

	def load(self) -> typing.List[Document]:
		blob = Blob.from_data(self.pBytes,path=self.file_path)
		docs=list(self.parser.parse(blob))
		return docs


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
		documents=None
		if 'file' in data:
			documents=PyPDFLoader(file_path=data['file'],extract_images=False).load()
		elif 'buffer' in data:
			documents=CustomPDFLoader(pBytes=bytes(data['buffer']),file_path=data['filename'] if 'filename' in data else '').load()
		else:
			pass

		if documents is not None:
			documents=[{"page_content":obj.page_content if hasattr(obj,'page_content') else "","metadata":obj.metadata if hasattr(obj,'metadata') else {}} for obj in documents]
			content=json.dumps({"state":"success","documents":documents})
			sys.stdout = old_stdout
			print(content+"\n",flush=True)
			sys.stdout = silent_stdout
		else:
			print('{"state":"error"}'+"\n",file=sys.__stderr__,flush=True)
	except BaseException as e:
		print(e,file=sys.__stderr__,flush=True)
