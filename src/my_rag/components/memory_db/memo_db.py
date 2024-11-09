from sklearn.decomposition import PCA
import torch


class MemoryDB:
    def __init__(self, similarity_threshold=0.45, fallback_threshold=3, max_memory_size=100, compression_ratio=8):
        self.short_term_memory = {}
        self.long_term_memory = {}
        self.similarity_threshold = similarity_threshold
        self.fallback_counter = {}
        self.fallback_threshold = fallback_threshold
        self.max_memory_size = max_memory_size
        self.memory = []
        self.access_counts = {}
        self.compressor = MemoryCompression(compression_ratio)

    def store_clues(self, doc_id, clues, memory_type="short"):
        memory = self.short_term_memory if memory_type == "short" else self.long_term_memory
        memory[doc_id] = clues

    def store_memory_tokens(self, doc_id, memory_tokens, memory_type="short"):
        memory = self.short_term_memory if memory_type == "short" else self.long_term_memory
        memory[doc_id] = memory_tokens
        print(f"Stored memory tokens for document ID: {doc_id} in {memory_type} memory")

    def retrieve_clues(self, query, memory_type="short"):
        memory = self.short_term_memory if memory_type == "short" else self.long_term_memory
        relevant_clues = []
        for clues in memory.values():
            for clue in clues:
                if query.lower() in clue.lower():
                    relevant_clues.append(clue)
        return relevant_clues or ["No relevant clues found."]

    @staticmethod
    def generate_clues(model, tokenizer, document_content):
        inputs = tokenizer.encode(document_content, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=128)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clues = generated_text.split('. ')
        print(f"Generated clues: {clues}")
        return clues

    @staticmethod
    def process_query(query):
        print(f"Querying external sources for: {query}")
        return "Simulated external document content relevant to the query."

    def find_matching_entry(self, query_embedding):
        if not self.memory:
            return None
        memory_embeddings = torch.stack([entry[0] for entry in self.memory])
        similarities = self.calculate_similarity(query_embedding, memory_embeddings)
        max_similarity, max_idx = similarities.max(dim=0)
        if max_similarity >= self.similarity_threshold:
            return max_idx.item()
        else:
            return None

    def provide_feedback(self, query_embedding, feedback_score=1.0):
        matching_idx = self.find_matching_entry(query_embedding)
        if matching_idx is not None:
            self.access_counts[matching_idx] += feedback_score

    @staticmethod
    def calculate_similarity(query, memory):
        query_norm = query / torch.norm(query, dim=-1, keepdim=True)
        memory_norm = memory / torch.norm(memory, dim=-1, keepdim=True)
        return torch.matmul(query_norm, memory_norm.T)

    def retrieve_top_k(self, query_embedding, k=5):
        query_key = str(query_embedding.tolist())
        dynamic_threshold = self.similarity_threshold

        if self.fallback_counter.get(query_key, 0) > self.fallback_threshold:
            dynamic_threshold = max(dynamic_threshold * 0.5, 0.3)  # Lower threshold further if needed

        if not self.memory:
            print("Memory is empty.")
            return [], []

        memory_embeddings = torch.stack([entry[0] for entry in self.memory])
        similarities = self.calculate_similarity(query_embedding, memory_embeddings)

        # Log similarities for debugging
        print(f"Similarity scores for query: {similarities.tolist()}")

        top_k_indices = torch.topk(similarities, k=k).indices
        top_k_docs = [(self.memory[idx][1], similarities[idx].item()) for idx in top_k_indices]

        # Check if the top document meets the dynamic threshold
        if top_k_docs and top_k_docs[0][1] >= dynamic_threshold:
            for idx in top_k_indices:
                self.access_counts[idx.item()] += 1
            return [doc for doc, _ in top_k_docs], [score for _, score in top_k_docs]

        # No similar question found, handling fallback
        print("No similar question found in MemoryDB. Processing with ChromaDB...(Component function)")

        # Fallback counter
        if query_key in self.fallback_counter:
            self.fallback_counter[query_key] += 1
        else:
            self.fallback_counter[query_key] = 1

        # Add to memory on excessive fallbacks
        if self.fallback_counter[query_key] >= self.fallback_threshold:
            answer_document = "Answer from ChromaDB"
            self.add_to_memory(query_embedding, answer_document)
            print(f"Question added to MemoryDB after {self.fallback_counter[query_key]} fallbacks.")

        return [], []

    def add_to_memory(self, embedding, document):
        if len(self.memory) >= self.max_memory_size:
            self._prune_memory()
        self.memory.append((embedding, document))
        self.access_counts[len(self.memory) - 1] = 0

    def _prune_memory(self):
        sorted_indices = sorted(self.access_counts, key=self.access_counts.get)
        num_to_prune = len(self.memory) // 10
        for idx in sorted_indices[:num_to_prune]:
            self.memory.pop(idx)
            del self.access_counts[idx.item()]


class MemoryCompression:
    def __init__(self, compression_ratio=8):
        self.compression_ratio = compression_ratio
        self.pca = PCA(n_components=compression_ratio)  # Initialize PCA with the desired number of components

    def compress_tokens(self, embeddings):
        # Fit PCA and transform embeddings
        compressed_embeddings = self.pca.fit_transform(embeddings)
        return compressed_embeddings


