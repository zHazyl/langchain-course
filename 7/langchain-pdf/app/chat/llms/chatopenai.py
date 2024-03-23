from langchain.chat_models import ChatOpenAI

def build_llm(chat_args, model_name="gpt-3.5-turbo"):
    return ChatOpenAI(streaming=chat_args.streaming, model_name=model_name)

def build_condense_question_llm():
    return ChatOpenAI(streaming=False)

def build_classify_llm():
    return ChatOpenAI()

def build_summary_llm():
    return ChatOpenAI(streaming=False, max_tokens=300)