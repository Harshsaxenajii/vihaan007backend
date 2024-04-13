from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Replicate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

app = Flask(__name__)

# Define function to create vector store
def create_vector_store(embeddings, text_chunks):
    # Your vector store creation code goes here...
    vector_store = ...
    return vector_store

# Your other functions and routes go here...
@app.route("/")
def home():
    return "hello"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required."}), 400

    chain = app.config.get('chain')
    history = app.config.get('history')

    if not chain or not history:
        return jsonify({"error": "Chain or history is not initialized."}), 500

    output = conversation_chat(query, chain, history)

    app.config['past'].append(query)
    app.config['generated'].append(output)

    return jsonify({"query": query, "response": output})

if __name__ == "__main__":
    text_chunks = []  # You need to define your text_chunks here

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = create_vector_store(embeddings, text_chunks)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    # Initialize session state
    app.config.update(initialize_session_state())

    app.config['chain'] = chain
    app.run(port=5000, debug=True)
