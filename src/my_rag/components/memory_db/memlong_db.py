from .base import BaseMemory
import torch


class MemoryDB(BaseMemory):
    def __init__(self, embedding_dim=512, max_memory_size=10000, similarity_threshold=0.8, fallback_threshold=3):
        super().__init__()
        self.memory = []  # Stores (embedding, document) tuples
        self.access_counts = {}  # Tracks access frequency for pruning
        self.fallback_counter = {}  # Tracks fallback occurrences
        self.max_memory_size = max_memory_size
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.fallback_threshold = fallback_threshold

    def add_to_memory(self, embedding, document):
        # If memory is full, prune the least frequently accessed entries
        if len(self.memory) >= self.max_memory_size:
            self._prune_memory()
        self.memory.append((embedding, document))
        self.access_counts[len(self.memory) - 1] = 1

    def calculate_similarity(self, query, memory):
        # Normalize for cosine similarity
        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        memory_norm = memory / torch.norm(memory, dim=-1, keepdim=True)
        return torch.matmul(query_norm, memory_norm.T)

    def retrieve_top_k(self, query_embedding, k=5):
        """Retrieve top-k documents, with fallback handling."""

        # Step 1: Calculate similarities with memory embeddings
        memory_embeddings = torch.stack([entry[0] for entry in self.memory])  # Only embeddings
        similarities = self.calculate_similarity(query_embedding, memory_embeddings)
        top_k_indices = torch.topk(similarities, k=k).indices
        top_k_docs = [(self.memory[idx][1], similarities[idx].item()) for idx in top_k_indices]

        # Step 2: Check if highest similarity meets the threshold
        if top_k_docs and top_k_docs[0][1] >= self.similarity_threshold:
            # Update access counts for pruning
            for idx in top_k_indices:
                self.access_counts[idx] += 1
            return [doc for doc, _ in top_k_docs]

        # Step 3: If no similar question found, increase fallback counter
        question_text = query_embedding.tolist()  # Using list representation for tracking (or use hash)
        question_key = str(question_text)
        if question_key in self.fallback_counter:
            self.fallback_counter[question_key] += 1
        else:
            self.fallback_counter[question_key] = 1

        # Print fallback message
        print("No similar question found in MemoryDB. Processing with ChromaDB...(Component function)")

        # Step 4: Check if fallback counter reached threshold
        if self.fallback_counter[question_key] >= self.fallback_threshold:
            # Add the current question-answer pair to MemoryDB
            # (Embedding should be generated when fallback answer is retrieved from ChromaDB)
            answer_document = "Answer from ChromaDB"  # Replace this with actual retrieval
            self.add_to_memory(query_embedding, answer_document)
            print(f"Question added to MemoryDB after {self.fallback_counter[question_key]} fallbacks.")

        return []  # Indicating to process with ChromaDB if no similar result found

    def _prune_memory(self):
        """Prune the least accessed entries in memory to maintain max_memory_size."""
        sorted_indices = sorted(self.access_counts, key=self.access_counts.get)
        num_to_prune = len(self.memory) // 10  # Prune the bottom 10% by access frequency
        for idx in sorted_indices[:num_to_prune]:
            self.memory.pop(idx)
            del self.access_counts[idx]
