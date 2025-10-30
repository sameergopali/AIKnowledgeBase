from langgraph.graph import StateGraph, START, END

from .state import State


from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from loguru import logger



class DocumentRelevanceScore(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class Suggestion(BaseModel):
    """Suggestions for improving document retrieval."""

    suggestions: list[str] = Field(
        description="Suggestions for documents, books, articles, or related topics to improve retrieval"
    )
    missing_info: list[str] = Field(
        description="Missing information or context that could improve retrieval"
    )

class ConfidenceScore(BaseModel):
    """Confidence score for the generated answer."""

    confidence: float = Field(
        description="Confidence score ranging from 0 to 1"
    )
    missing_info: list[str] = Field(
        description="Missing information or context that could improve answer generation"
    )
    suggestions: list[str] = Field(
        description="Suggestions for improving answer generation"
    )
    
class SuggestionRAGWorkflow:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.graph = self.build()
    
    def retrieve(self, state: State) -> State:
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        documents = self.retriever.retrieve(question, n_results=5, rerank_top_k=3)
        return {"documents": documents}

    
    def check_documents(self, state: State) -> State:
        print("---CHECK DOCUMENTS---")
        if len(state["documents"])==0:
            return {"relevant": 'no'}
        context = "\n\n".join(doc["page_content"] for doc in state["documents"])
        template = PromptTemplate.from_template(
            "Document:\n{context}\n\nQuestion: {question}"
        )
        prompt = template.format(context=context, question=state["question"])
        llm = self.llm.with_structured_output(DocumentRelevanceScore)
        response = llm.invoke([
            {"role": "system", "content": """You are a grader assessing relevance of a retrieved document to a user question. \n
                If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""},
            {"role": "user", "content": prompt}])
        print("relevant docs:", response.binary_score)
        return {"relevant": response.binary_score}
    
    
    def decide_to_generate(self, state):
        print("---ASSESS GRADED DOCUMENTS---")
        relevant = state["relevant"]
        if relevant == "no":
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
            )
            return "suggest_enrichment"
        else:
            print("---DECISION: GENERATE---")
            return "generate"
    
    
        
    def generate(self, state: State) -> State:
        print("---GENERATE ANSWER---")
        context = "\n\n".join(doc["page_content"] for doc in state["documents"])
        template = PromptTemplate.from_template("""Context:
                    {context}   
                    Question: {question}""")
        prompt = [{"role": "system", "content": """Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Use three sentences maximum and keep the answer as concise as possible.
                    Always say "thanks for asking!" at the end of the answer."""},
                  {"role": "user", "content": template.format(context=context, question=state["question"])}]
        answer = self.llm.invoke(
            prompt
        )
        
        return {"answer": answer.content}
    
    def suggest_enrichment(self, state)-> State:
        print("---SUGGEST ENHANCEMENT---")
        template = PromptTemplate.from_template("""Question: {question}""")
        prompt = [{"role": "system", "content": """You are an expert at enriching knowledge base. You provide suggestions and missing information for given query.
                    Provide the following:
                    Missing or Uncertain Information:
                    1. Specify which facts, concepts, or perspectives are missing.
                    2. Highlight ambiguities or uncertainties in the document that limit a complete answer.
                    Enrichment Suggestions:
                    1. Recommend up to three additional sources, topics, or data types that would help fill the missing gaps or improve retrieval quality.
                    2. Keep suggestions specific and actionable (e.g., “Add documentation on AWS SES inbound email processing,” not “find more info about AWS”)"""},
                  {"role": "user", "content": template.format(question=state["question"])}]
        llm = self.llm.with_structured_output(Suggestion)
        answer = llm.invoke(
            prompt
        )
        return {"answer": "Sorry, Relevant document found to answer the query. Check answer details for suggestions", "suggestions":answer.suggestions, "missing_info":answer.missing_info, "confidence": 0.0 }
            
    def check_confidence(self, state):
        print("---CHECK CONFIDENCE---")
        context = "\n\n".join(doc["page_content"] for doc in state["documents"])
        question = state["question"]
        answer = state["answer"]
        template = PromptTemplate.from_template("""Contex: {context} \n\n Question: {question}\n\n Answer:{answer}""")
        prompt = [{"role": "system", "content": """
                    You are AnswerEvaluatorAI, an expert system for evaluating the accuracy and completeness of answers based on provided documents.
                    Your Objective: 
                    1. Assess whether the given Answer fully and correctly addresses the Question, using evidence from the Document.
                    2. Determine if the answer is factually accurate and fully supported by the content of the provided document.
                    3. Check for coverage completeness — does the answer address all key aspects of the question that are present or inferable from the document?
                    4. Identify any irrelevant, unsupported, or hallucinated claims.
                    Confidence Scoring:
                    1. Output a confidence score between 0 and 1, indicating how certain you are that the answer is accurate and complete.
                        1.0 = fully correct and complete
                        0.0 = inaccurate or entirely unsupported
                    Missing or Uncertain Information:
                    1. If the answer is incomplete, specify which facts, concepts, or perspectives are missing.
                    2. Highlight ambiguities or uncertainties in the document that limit a complete answer.
                    Enrichment Suggestions
                    1. Recommend up to three additional sources, topics, or data types that would help fill the missing gaps or improve retrieval quality.
                    2. Keep suggestions specific and actionable (e.g., “Add documentation on AWS SES inbound email processing,” not “find more info about AWS”)
                    """},
                  {"role": "user", "content": template.format(context=context, question=question, answer=answer)}]
        llm = self.llm.with_structured_output(ConfidenceScore)
        checked = llm.invoke(
            prompt
        )
        return {"suggestions":checked.suggestions, "missing_info":checked.missing_info , "confidence":checked.confidence}
        
    
        
    def build(self):
        workflow = StateGraph(State)
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.check_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("suggest_enrichment", self.suggest_enrichment)  # web search node (reuse generate)
        workflow.add_node("check_confidence", self.check_confidence)
        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "suggest_enrichment": "suggest_enrichment",
                "generate": "generate",
            },
        )
        workflow.add_edge("suggest_enrichment", END)
        workflow.add_edge("generate", "check_confidence")
        workflow.add_edge("check_confidence", END)
        return workflow.compile()


    def invoke(self, question: str) -> State:
        initial_state = State(question=question)
        final_state = self.graph.invoke(initial_state)
        return final_state