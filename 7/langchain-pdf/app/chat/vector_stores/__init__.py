from functools import partial
from .pinecone import build_retriever, vector_store

retriever_map = {
    "pinecone_2": partial(build_retriever, k=1),
    "pinecone_2": partial(build_retriever, k=2),
    "pinecone_3": partial(build_retriever, k=3)
}