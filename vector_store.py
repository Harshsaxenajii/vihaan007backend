# Create vector store
from f_app import embeddings, text_chunks


from langchain_community.vectorstores import FAISS


vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)