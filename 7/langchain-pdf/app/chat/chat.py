from app.chat.models import ChatArgs
from app.chat.vector_stores.pinecone import build_retriever, build_summary_retriever
from app.chat.llms.chatopenai import build_llm , build_condense_question_llm, build_summary_llm
from app.chat.llms import gemini
from app.chat.memories.sql_memory import build_memory, build_summary_memory
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.chains.retrieval_qa import StreamingRetrievalQA
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    retriever = build_retriever(chat_args)
    llm = build_llm(chat_args)
    condense_question_llm = gemini.build_condense_question_llm()
    # condense_question_llm = build_condense_question_llm()

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its English.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    memory = build_memory(chat_args, k=4)
    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=condense_question_llm,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT
    )

def build_summary(chat_args):
    retriever = build_summary_retriever(chat_args)
    combine_llm = build_llm(chat_args)
    summary_llm = gemini.build_summary_llm()
    memory = build_summary_memory(chat_args, k=0)
    summary_template = """
        Use English, according the following content summarize related to the user request "{question}":
        {summaries}
        =========
        SUMMARY:
    """
    question_prompt_template = """
        Use the following portion of a long document to summarize:
        {context}
        Summary:
    """
    return StreamingRetrievalQA.from_chain_type(
        llm=summary_llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "reduce_llm": combine_llm,
            "combine_prompt": PromptTemplate(
                template=summary_template, input_variables=["summaries"]
            ),
            "question_prompt": PromptTemplate(
                template=question_prompt_template, input_variables=["context"]
            ),
            "memory": memory
        }
    )