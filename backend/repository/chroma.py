from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
from sentence_transformers import CrossEncoder
import uuid
from loguru import logger


class ChromaDB:
    """
    A singleton manager class for ChromaDB collections used in RAG pipelines.
    Includes reranking to improve relevance after vector similarity search.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        collection_name: str = "default_collection",
        persist_dir: Optional[str] = "./chroma_data",
        embedding_model: Optional[Any] = embedding_functions.DefaultEmbeddingFunction(),
        reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            collection_name: Name of the collection to connect to or create.
            persist_dir: Directory to persist data. If None, runs in-memory.
            embedding_model: Optional custom embedding model (e.g., SentenceTransformer).
            reranker_model_name: Optional reranker model (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
        """
        # Only initialize once
        if self._initialized:
            return
        
        settings = (
            Settings(persist_directory=persist_dir, is_persistent=True) if persist_dir else Settings()
        )
        self.client = Client(settings)
        self.embedding_fn = embedding_model
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        
        # Optional cross-encoder reranker
        self.reranker = (
            CrossEncoder(reranker_model_name)
            if reranker_model_name is not None
            else None
        )
        
        self._initialized = True


    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """Add or upsert documents and their embeddings into the Chroma collection."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(documents)} documents to collection '{self.collection.name}'")
        return ids

  

    def delete(self, ids: List[str]):
        """Delete documents by ID."""
        self.collection.delete(ids=ids)

    def reset_collection(self):
        """Clear all data in the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
        )

    def persist(self):
        """Persist changes to disk."""
        # Persisting is handled by the client configuration, this is a placeholder
        logger.info("Persistence is managed by the client settings.")
        pass


    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"category": "science"})
        
        Returns:
            Dict with ids, documents, metadatas, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        result = {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
        }
        if self.reranker is None or rerank_top_k is None:
            return result
        # Rerank top K results
        logger.info(f"Reranking top {rerank_top_k} results using cross-encoder.")
        pairs = [(query, doc) for doc in result["documents"][:rerank_top_k]]
        scores = self.reranker.predict(pairs)
        logger.info(f"Reranker scores: {scores}")
        reranked_results = sorted(
            zip(scores, result["ids"][:rerank_top_k], result["documents"][:rerank_top_k], result["metadatas"][:rerank_top_k]),
            key=lambda x: x[0],
            reverse=False
        )

        return {
            "ids": [item[1] for item in reranked_results] ,
            "documents": [item[2] for item in reranked_results] ,
            "metadatas": [item[3] for item in reranked_results] ,
        }
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        rerank_top_k: Optional[int] = None,
    ) -> list:
        """
        Retrieve documents similar to a query.

        Args:
            query: The search query string.
            n_results: The number of documents to return.
            where: Optional metadata filter.
            rerank_top_k: Number of top results to rerank.

        Returns:
            A list of LangChain Document objects.
        """
        search_results = self.similarity_search(
            query=query,
            n_results=n_results,
            where=where,
            rerank_top_k=rerank_top_k,
        )

        documents = []
        for i, doc_content in enumerate(search_results["documents"]):
            document = {
                    "page_content": doc_content,
                    "metadata": search_results["metadatas"][i]
            }
            documents.append(document)
            
        return documents