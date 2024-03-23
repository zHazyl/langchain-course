from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    vectorstore: VectorStore

    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
            k=6
        )

    async def aget_relevant_documents(self):
        return []
