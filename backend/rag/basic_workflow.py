from langgraph.graph import StateGraph, START, END

from .state import State


from langchain_core.prompts import PromptTemplate

SYSTEM= """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer."""

USER = """Context:
{context}   
Question: {question}"""




class BasicRAGWorkflow:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.graph = self.build()
    
    def retrieve(self, state: State) -> State:
        question = state["question"]
        documents = self.retriever.retrieve(question, n_results=5, rerank_top_k=3)
        return {"documents": documents}

    def generate(self, state: State) -> State:
        context = "\n\n".join(doc["page_content"] for doc in state["documents"])
        template = PromptTemplate.from_template(USER)
        prompt = [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": template.format(context=context, question=state["question"])}]
        answer = self.llm.invoke(
            prompt
        )
        return {"generation": answer}
        
    def build(self):
        workflow = StateGraph(State)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)
        
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()
        

    def invoke(self, question: str) -> State:
        initial_state = State(question=question)
        final_state = self.graph.invoke(initial_state)
        print(final_state)
        return final_state
 