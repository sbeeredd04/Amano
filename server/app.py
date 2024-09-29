from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from invokeLLM import invokeLLM

app = Flask(__name__)

CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Welocme to Jodi"})

@app.route('/chatbot')
def chatbot():
    input = request.form.get("input")
    response = invokeLLM(input)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)