module.exports = function(RED){
	function rFCNode(config){
		const path = require('path')
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/rag-pdf-loader.py'
		node.config = {
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("rag-pdf-loader", rFCNode);
}
