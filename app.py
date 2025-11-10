"""
Flask web application for Olist RAG Assistant
"""

from flask import Flask, render_template, request, jsonify
from rag_app import OlistRAGAssistant, FREE_MODELS
import os

app = Flask(__name__)

# Global assistant instance
assistant = None


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models=FREE_MODELS)


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG assistant with API key"""
    global assistant
    
    data = request.json
    api_key = data.get('api_key')
    model_name = data.get('model_name', 'meta-llama/llama-3.2-3b-instruct:free')
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
    
    try:
        assistant = OlistRAGAssistant(api_key=api_key, model_name=model_name)
        return jsonify({'status': 'success', 'message': 'Assistant initialized successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask():
    """Ask a question to the RAG assistant"""
    global assistant
    
    if assistant is None:
        return jsonify({'error': 'Assistant not initialized. Please provide API key first.'}), 400
    
    data = request.json
    question = data.get('question')
    top_k = data.get('top_k', 5)
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        result = assistant.ask(question, top_k=top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available free models"""
    return jsonify({'models': FREE_MODELS})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

