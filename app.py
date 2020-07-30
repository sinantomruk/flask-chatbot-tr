from flask import Flask, render_template, request
from botfuncs import chatbot_response
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM
import os

app = Flask(__name__)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if(userText == "quit"):
        shutdown_server()
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()

#shutdownJVM()
