# node-red-contrib-ml-rag
This module for Node-RED contains a set of nodes which offer machine learning functionalities for Retrieval Augmented Generation (RAG) tasks.


## Pre requisites
Be sure to have a working installation of [Node-RED](https://nodered.org/ "Node-RED").  
Install python and the following libraries:
* [Python](https://www.python.org/ "Python") 3.9.+ accessible by the command 'python' (on linux 'python3')
* Full pip install: pip install langchain langchain_core PyPDF2 langchain-community pypdf


## Install
To install the latest version use the Menu - Manage palette option and search for node-red-contrib-ml-rag, or run the following command in your Node-RED user directory (typically ~/.node-red):

    npm i node-red-contrib-ml-rag

## Usage
These flows create a dataset, train a model and then evaluate it. Models, after training, can be use in real scenarios to make predictions.
Dataset must contain 'text' (input) and 'label' (target) columns.

rag-embedding-generator: you can use models from here: https://huggingface.co/models?library=sentence-transformers

**Tip:** you can run 'node-red' (or 'sudo node-red' if you are using linux) from the folder '.node-red/node-modules/node-red-contrib-ml-rag' and the paths will be automatically correct.

This flow loads a training partition and trains a 'text-classify-trainer', saving the model locally.
![Training](https://i.imgur.com/oIDHwYu.png "Training")

This flow loads a test partition and evaluates a previously trained model.
![Evaluation](https://i.imgur.com/ufHBYLx.png "Evaluation")

You can use text classification model from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending "Hugging Face")

Example flows available here:
```json
[
  
]
```
## Thanks
Thanks to  Gabriele Maurina for awesome nodes - [node-red-contrib-machine-learning](https://github.com/GabrieleMaurina/node-red-contrib-machine-learning "node-red-contrib-machine-learning") 