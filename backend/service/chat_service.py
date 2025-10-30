from rag import BasicRAGWorkflow, SuggestionRAGWorkflow, SearchRAGWorkflow

class ChatService:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        self.suggestion_rag = SuggestionRAGWorkflow(retriever=db, llm=llm)
        self.search_rag = SearchRAGWorkflow(retriever=db, llm=llm)
        self.chat_history = []

    def chat(self, message: str, rag) -> str:
        print("ChatService received message:", message)
        if rag == "suggestion-rag":
            response = self.suggestion_rag.invoke(message)
        elif rag == "search-rag":
            response = self.search_rag.invoke(message)
        self.chat_history.append({"role":"user", "content":message})
        self.chat_history.append({"role":"ai", "content":response["answer"]})

        return response