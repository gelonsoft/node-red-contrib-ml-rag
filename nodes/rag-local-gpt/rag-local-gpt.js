module.exports = function(RED){
	function rFCNode(config){
		const utils = require('../../utils/utils')

		const node = this;

		//set configurations
		node.file = __dirname +  '/rag-local-gpt.py'
		node.config = {
			modelNameOrPath: config.modelNameOrPath || 'openai-community/gpt2',
			promptTemplate: config.promptTemplate || "Текст: {context}\n\nВопрос: {input}",
			deviceMap: config.deviceMap || 'auto',
			maxNewTokens: ~~config.maxNewTokens || 2048,
			doSample: config.doSample===1 || 1,
			topK: Number.parseFloat(config.topK) || 0,
			topP: Number.parseFloat(config.topP) || 0.15,
			temperature: Number.parseFloat(config.temperature) || 0.3,
			repetitionPenalty: Number.parseFloat(config.repetitionPenalty) || 1.1
		}

		utils.run(RED, node, config)
	}
	RED.nodes.registerType("rag-local-gpt", rFCNode);
}
