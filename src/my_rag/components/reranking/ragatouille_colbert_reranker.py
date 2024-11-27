from .base import BaseReranker
from typing import List, Dict, Any
from ragatouille import RAGPretrainedModel


class ColBERTReranker(BaseReranker):
    """Reranker implementation using ColBERT"""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model = RAGPretrainedModel.from_pretrained(model_name)

    def rerank(self, query: str, documents: List[str], k: int) -> List[Dict[str, Any]]:
        return self.model.rerank(query=query, documents=documents, k=k)
