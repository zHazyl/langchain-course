from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI

def build_classify_llm(model_name="gemini-pro"):
    return GoogleGenerativeAI(model=model_name)

def build_condense_question_llm(model_name="gemini-pro"):
    return ChatGoogleGenerativeAI(streaming=False, model=model_name, temperature=0)

def build_summary_llm(model_name="gemini-pro"):
    return ChatGoogleGenerativeAI(streaming=False, model=model_name, convert_system_message_to_human=True, max_output_tokens=300)