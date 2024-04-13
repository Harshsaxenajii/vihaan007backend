from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile

app = Flask(__name__)

load_dotenv()

def create_conversational_chain(vector_store):
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        input={"temperature": 0.01, "max_length": 4500, "top_p": 1}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json['query']
    history = request.json.get('history', [])
    output = conversation_chat(query, chain, history)
    return jsonify({"response": output, "history": history + [(query, output)]})

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    return result["answer"]

if __name__ == "__main__":
    uploaded_files = []  # Assuming files will be uploaded before running the Flask app

    text = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    chain = create_conversational_chain(vector_store)

    app.run(debug=True)  # Run Flask app in debug mode for development
