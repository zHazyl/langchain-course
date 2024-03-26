from typing import Any, Optional, Union
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
from app.chat.vector_stores import vector_store
from app.web.db.models.pdf import Pdf

class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.streaming_run_ids = set()
        self.human_message = {}
    
    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        if serialized["kwargs"]["streaming"]:
            self.streaming_run_ids.add(run_id)
            self.human_message[run_id] = messages[-1][1].content

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)
    
    def on_llm_end(self, response: LLMResult, run_id, **kwargs):
        if run_id in self.streaming_run_ids:
            self.queue.put("\n--------------Source--------------")
            for vector in vector_store.similarity_search(k=2, query=self.human_message[run_id]):
                pdf_id = vector.metadata['pdf_id']
                self.queue.put("\n[" + Pdf.find_by(id=pdf_id).as_dict()['name'] + f" [page {vector.metadata['page']}]]\n\"{vector.metadata['text']}\"")
                
            while True:
                if self.queue.empty():
                    break
            self.queue.put(None)
            self.streaming_run_ids.remove(run_id)
            self.human_message.__delitem__(run_id)

    def on_llm_error(self, error, **kwargs):
        self.queue.put(None)