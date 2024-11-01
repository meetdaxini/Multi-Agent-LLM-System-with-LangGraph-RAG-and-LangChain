from .base import BaseMemory
import torch

class MemoryDB(BaseMemory):
    def __init__(self, embedding_dim=512, max_memory_size=10000):
        super().__init__()
        self.memory = []  # Stores (embedding, document) tuples
        self.access_counts = {}
        self.max_memory_size = max_memory_size
        self.embedding_dim = embedding_dim

    def add_to_memory(self, embedding, document):
        if len(self.memory) >= self.max_memory_size:
            self._prune_memory()
        self.memory.append((embedding, document))
        self.access_counts[len(self.memory) - 1] = 1

    def retrieve_top_k(self, query_embedding, k=5):
        if len(self.memory) == 0:
            return []
        memory_embeddings = torch.stack([entry[0] for entry in self.memory])
        similarities = self.calculate_similarity(query_embedding, memory_embeddings)
        top_k_indices = torch.topk(similarities, k=k).indices
        retrieved_docs = [self.memory[idx][1] for idx in top_k_indices]
        for idx in top_k_indices:
            self.access_counts[idx] += 1
        return retrieved_docs

    def _prune_memory(self):
        sorted_indices = sorted(self.access_counts, key=self.access_counts.get)
        num_to_prune = len(self.memory) // 10
        for idx in sorted_indices[:num_to_prune]:
            self.memory.pop(idx)
            del self.access_counts[idx]
