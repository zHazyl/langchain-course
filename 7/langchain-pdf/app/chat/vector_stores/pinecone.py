import os
import pinecone
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings
from langchain.vectorstores.chroma import Chroma
 
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENV_NAME")
# )
 
# vector_store = Pinecone.from_existing_index(
#     os.getenv("PINECONE_INDEX_NAME"), embeddings
# )

pinecone.init(
    api_key='2f1c9240-3c15-4d50-b75e-a252d5769c3a',
    environment='gcp-starter'
)

vector_store = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)