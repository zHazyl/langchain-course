from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.schema import BaseRetriever
from threading import Thread
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document

class ManualRetriever(BaseRetriever):
    embeddings: Embeddings
    vectorstore: VectorStore

    def get_relevant_documents(self, query):
        # calculate embeddings for the 'query' string
        # emb = self.embeddings.embed_query(query)

        context = self.answer_question(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vector
        return [Document(page_content=context)]

    async def aget_relevant_documents(self):
        return []
    
    def answer_question(self, input):
        questions_prompt = f"""
        You're a good assistant that support me ask questions with different perspective to answer my question better.
        For example:

        Question: How can I open a company?
        Answer: Which laws you need to know?++What skills you need to prepare?++Which procedure you need to make?

        Question: How can I make udon?
        Answer: What are ingredients to make udon?++Where to find ingredients to make udon?++What recipe to make udon?
        =========
        Question: {input}
        Answer:
        """

        retriever = self.vectorstore.as_retriever(search_kwargs={'k': 1})

        gpt = RetrievalQA.from_llm(
            llm=ChatOpenAI(),
            retriever=retriever
        )
        gemini = RetrievalQA.from_llm(
            llm=ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True),
            retriever=retriever
        )

        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        llms = [None, gpt, gemini]
        index = -1

        questions = llm.invoke(questions_prompt).content
        questions = list(filter(lambda q: len(q) != 0, questions.split("++")))
        # questions.append(input)
        print(questions)

        ans = []

        for question in questions:
            t = Thread(target=lambda q: ans.append(llms[index].invoke(q)['result']), args=[question])
            t.start()
            index*=-1

        while True:
            if len(ans) == len(questions):
                return '\n-'.join(ans)
