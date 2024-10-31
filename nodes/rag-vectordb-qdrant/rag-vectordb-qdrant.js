module.exports = function(RED){
	function rFCNode(config){
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/rag-vectordb-qdrant.py'
		node.config = {
			localSavePath: config.localSavePath || '/tmp/qdrant',
			remoteUrl: config.remoteUrl || '',
			remoteApiKey: config.remoteApiKey || '',
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("rag-vectordb-qdrant", rFCNode);
}
