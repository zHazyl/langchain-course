import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings
from .redundant_filter_retriever import RedundantFilterRetriever
from manual_retriever import ManualRetriever
 
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV_NAME")
)

# pinecone.Index(os.getenv("PINECONE_INDEX_NAME")).delete(delete_all=True, namespace="")
 
# vector_store = Pinecone.from_existing_index(
#     os.getenv("PINECONE_INDEX_NAME"), embeddings
# )

from langchain.vectorstores.chroma import Chroma
vector_store = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

def build_retriever(chat_args, k=2):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, "k": k}
    return vector_store.as_retriever(
        # search_kwargs=search_kwargs
        k=2
    )

def build_manual_retriever():
    return ManualRetriever(embeddings=embeddings, vectorstore=vector_store)

def build_summary_retriever(chat_args, k=4):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, "k": k}
    return RedundantFilterRetriever(
        embeddings=embeddings,
        vectorstore=vector_store,
        # search_kwargs=search_kwargs
    )