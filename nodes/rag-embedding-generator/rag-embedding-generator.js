module.exports = function(RED){
	function rFCNode(config){
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/rag-embedding-generator.py'
		node.config = {
			modelNameOrPath: config.modelNameOrPath || 'sergeyzh/LaBSE-ru-turbo',
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("rag-embedding-generator", rFCNode);
}
