from langchain.chains import RetrievalQA
from app.chat.chains.streamable import StreamableChain

class StreamingRetrievalQA(
    StreamableChain, RetrievalQA
):
    pass