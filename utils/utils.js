const status = require('./status.js')
const {spawn} = require('child_process')
const { Buffer } = require('node:buffer');

//use 'python3' on linux and 'python' on anything else
const pcmd = process.platform === 'linux' ? 'python3' : 'python'

//initialize child process
const initProc = (node) => {
    if (node.proc == null) {
        node.proc = spawn(pcmd, [node.file], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {...process.env, PYTHONUNBUFFERED: "1"}
        })

        //handle results
        let stdOutData=''
        node.proc.stdout.on('data', (data) => {
            const dataStr=data.toString()
            stdOutData+=data.toString()
            if (dataStr.indexOf("\n")!==-1) {
                node.status(status.DONE)
                //console.log("stdout data",data)
                let newPayload
                try {
                    newPayload = JSON.parse(stdOutData)
                } catch (err) {
                    newPayload = stdOutData
                }
                if (newPayload!=="\n" && newPayload!=="\r\n" && newPayload!==undefined && newPayload!==null) {
                    node.msg.payload=newPayload
                    var msg = node.msg
                    if (node.wires.length > 1) {
                        msg = [node.msg, null]
                    }
                    node.send(msg)
                }
                stdOutData=""
            }
        })


        //handle errors
        node.proc.stderr.on('data', (data) => {
            node.status(status.ERROR)
            console.error("Error", data.toString())
            try {
                node.msg.payload = JSON.parse(data.toString())
            } catch (err) {
                node.msg.payload = data.toString()
            }
            var msg = node.msg
            if (node.wires.length > 1) {
                msg = [null, node.msg]
            }
            node.send(msg)
        })

        //handle crashes
        node.proc.on('exit', () => {
            console.log("subprocess exit")
            node.proc = null
        })

        //send node configurations to child
        node.proc?.stdin?.write(Buffer.from(JSON.stringify(node.config)).toString('base64') + "\t\t\t\n")
    }
}

//send payload as json to python script
const python = (node) => {
    initProc(node)
    node.proc?.stdin?.write(Buffer.from(JSON.stringify(node.msg.payload)).toString('base64') + "\t\t\t\n")
}

module.exports = {
    python: python,
    //parse string containing comma separated integers
    listOfInt: (str) => {
        var ints = null
        try {
            ints = str.replace(' ', '').split(',').map((n) => parseInt(n))
            if (ints.some(isNaN)) {
                ints = null
            }
        } finally {
            return ints
        }
    },

    //initialize node
    run: (RED, node, config) => {
        RED.nodes.createNode(node, config)
        node.status(status.NONE)

        node.proc = null
        node.msg = {}
        initProc(node)

        //process message
        const handle = (msg) => {
            node.status(status.PROCESSING)
            node.msg = msg
            if (node.topic != undefined) {
                node.msg.topic = node.topic
            }
            //send to python child
            python(node)
        }

        //handle input
        node.on('input', (msg) => {
            //if the node requires preprocessing of message, call preMsg
            if (node.preMsg != undefined) {
                node.preMsg(msg, handle)
            } else {
                handle(msg)
            }
        })

        //when node is closed, kill child process
        node.on('close', (done) => {
            node.status(status.NONE)
            if (node.proc != null) {
                node.proc.kill()
                node.proc = null
            }
            done()
        })
    }
}
