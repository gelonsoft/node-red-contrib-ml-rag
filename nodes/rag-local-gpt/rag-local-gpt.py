import sys

old_stdout=sys.__stdout__
silent_stdout = sys.__stderr__
sys.stdout = silent_stdout

import traceback
import typing
import json
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from urllib.parse import unquote
from langchain_huggingface import  HuggingFacePipeline
import os
import base64

if os.environ.get('RAG_DISABLE_SSL_VERIFY', "0") == "1":
	print("Disabling ssl verify")
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	#os.environ['REQUESTS_CA_BUNDLE'] = 'somepath/rootca.crt'

class PromptTemplateCutContextToSize(PromptTemplate):
	max_length: int =2048
	llm:HuggingFacePipeline=None
	def set_llm(self,p_max_length,p_llm):
		self.llm=p_llm
		self.max_length=p_max_length
	def format(self, **kwargs: typing.Any) -> str:
		old_context=kwargs.get('context',"")
		old_input=kwargs.get('input',"")
		message=super().format(**kwargs)
		num_tokens=self.llm.get_num_tokens(message)
		while int(num_tokens)>self.max_length:
			if len(old_context) > 0:
				old_context=old_context.rsplit(' ', 1)[0]
				kwargs.update(context=old_context)
				message=super().format(**kwargs)
				num_tokens=self.llm.get_num_tokens(message)
			elif len(old_input)> 0:
				old_input=old_input.rsplit(' ', 1)[0]
				kwargs.update(input=old_input)
				message=super().format(**kwargs)
				num_tokens=self.llm.get_num_tokens(message)
			else:
				break
		return message

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

context={'document_chain':None}

def create_document_chain(p_config):
	model_llm = AutoModelForCausalLM.from_pretrained(p_config['modelNameOrPath'])
	model_llm_input_size=model_llm.lm_head.in_features
	tokenizer_llm = AutoTokenizer.from_pretrained(p_config['modelNameOrPath'], max_length=model_llm_input_size, truncation=True)
	tokenizer_llm.pad_token = tokenizer_llm.eos_token
	pipe = pipeline("text-generation",
					model=model_llm, tokenizer=tokenizer_llm,
					device_map=p_config['deviceMap'],
					max_new_tokens=p_config['maxNewTokens'],
					do_sample = p_config['doSample']==1,
					top_k=p_config['topK'],
					top_p = p_config['topP'],
					temperature=p_config['temperature'],
					repetition_penalty=p_config['repetitionPenalty'])
	llm = HuggingFacePipeline(pipeline=pipe)
	prompt=PromptTemplateCutContextToSize(input_variables=["input","context"],template=p_config['promptTemplate'])
	prompt.set_llm(model_llm_input_size,llm)
	document_chain = create_stuff_documents_chain(llm, prompt)
	return (document_chain,prompt)

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
		#Lazy load
		if context['document_chain'] is None:
			doc_chain=create_document_chain(config)
			context['document_chain']=doc_chain[0]
			context['prompt']=doc_chain[1]
			doc_chain=None
		#config reload in runtime
		if 'config' in data:
			data_conf=data['config']
			if 'modelNameOrPath' in data_conf:
				config['modelNameOrPath']=data_conf['modelNameOrPath']
			if 'promptTemplate' in data_conf:
				config['promptTemplate']=data_conf['promptTemplate']
			doc_chain=create_document_chain(config)
			context['document_chain']=doc_chain[0]
			context['prompt']=doc_chain[1]
			doc_chain=None
		if 'documents' in data and 'input' in data:
			documents=data['documents']
			documents=[Document(page_content=obj['page_content'] if 'page_content' in obj else '') for obj in documents]
			if 'prompt_only' in data and data['prompt_only']==1:
				res=context['prompt'].invoke({"context": documents, "input": data['input']})
			else:
				res=context['document_chain'].invoke({"context": documents, "input": data['input']})
			content=json.dumps({"state":"success","result":res})
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
