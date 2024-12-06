from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    """Abstract base class for reranking implementations"""

    @abstractmethod
    def rerank(self, query: str, documents: List[str], k: int) -> List[Dict[str, Any]]:
        """Reranks documents based on query"""
        pass
