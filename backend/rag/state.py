from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain.messages import AnyMessage

class State(TypedDict):
    question: str
    answer: str
    documents: List[Document]
    confidence: float
    suggestions: list[str]
    missing_info: list[str]
    

    
