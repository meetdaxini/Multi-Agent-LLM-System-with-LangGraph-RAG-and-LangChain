# base.py
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """Base class providing shared utility methods for memory operations."""

    def calculate_similarity(self, query_embedding, memory_embeddings):
        """Calculate cosine similarity between query and stored embeddings."""
        return F.cosine_similarity(query_embedding, memory_embeddings)

    @abstractmethod
    def add_to_memory(self, embedding, document):
        pass

    @abstractmethod
    def retrieve_top_k(self, query_embedding, k=5):
        pass

    @abstractmethod
    def _prune_memory(self):
        pass
