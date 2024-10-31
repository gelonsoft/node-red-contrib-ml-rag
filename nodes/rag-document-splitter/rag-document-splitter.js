module.exports = function(RED){
	function rFCNode(config){
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/rag-document-splitter.py'
		node.config = {
			chunkSize: config.chunkSize || 2048,
			chunkOverlap: config.chunkOverlap || 0,
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("rag-document-splitter", rFCNode);
}
