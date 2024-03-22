from langchain_google_genai import GoogleGenerativeAI

def build_classify_llm(model_name="models/gemini-1.0-pro"):
    return GoogleGenerativeAI(model=model_name)