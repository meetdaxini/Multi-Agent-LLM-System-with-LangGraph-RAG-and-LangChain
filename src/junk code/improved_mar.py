from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
import time
import torch
import numpy as np
from typing import List, Tuple, Optional
import logging
from torch.nn import functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    embedding: np.ndarray
    document: str
    access_count: int
    last_access: float
    score: float = 0.0


# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Define main and memory databases (collections in ChromaDB)
main_db = client.get_or_create_collection("main_db")  # Collection for main storage
memory_db = client.get_or_create_collection("memory_db")  # Collection for memory storage


# Define a function to query the memory database first, then the main database
def process_query(query_embedding):
    # Step 1: Query Memory DB
    memory_results = memory_db.query(
        query_embeddings=[query_embedding],
        n_results=5  # Return top 5 most similar results
    )

    # Check if results were found in Memory DB
    if memory_results["documents"]:
        return memory_results["documents"]

    # Step 2: Query Main DB if no results in Memory DB
    main_results = main_db.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    return main_results["documents"] if main_results["documents"] else "No results found."


class ChromaMemoryDB:
    def __init__(
            self,
            collection_name: str = "memory_store",
            embedding_dim: int = 512,
            similarity_threshold: float = 0.8,
            cache_size: int = 1000,
            max_memory_size: int = 10000
    ):
        # Initialize ChromaDB client
        self.client = chromadb.Client()  # Use in-memory client

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection '{collection_name}'.")
        except Exception:
            logger.info(f"No existing collection '{collection_name}' to delete, starting fresh.")

        # Create a new collection
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.info(f"Created ChromaDB collection (database) with name: {collection_name}")

        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_memory_size = max_memory_size
        self.cache = {}
        self.cache_size = cache_size
        self.access_stats = {}


    def add_to_memory(self, query_embedding: torch.Tensor, document: str):
        """Add document to memory using ChromaDB."""
        # Convert embedding to numpy
        embedding_np = query_embedding.cpu().numpy()

        # Generate a unique ID for the document
        doc_id = str(int(time.time() * 1000))

        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding_np.tolist()],
            documents=[document],
            ids=[doc_id],
            metadatas=[{"timestamp": str(time.time()), "access_count": 1}]
        )

        # Add to cache
        self._update_cache(doc_id, embedding_np, document)

        # Initialize access stats
        self.access_stats[doc_id] = {
            "count": 1,
            "last_access": time.time()
        }

        # Check if we need to prune
        if self.collection.count() > self.max_memory_size:
            self._prune_memory()

    def retrieve_top_k(self, query_embedding: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve similar documents using ChromaDB with cache support."""
        # Convert query to numpy
        query_np = query_embedding.cpu().numpy()

        # Check cache first
        cache_results = self._check_cache(query_embedding)
        if cache_results:
            return cache_results

        # Query ChromaDB
        k = max(1, k)
        n_results = max(1, min(k * 2, self.collection.count()))  # Ensure n_results is also at least 1
        results = self.collection.query(
            query_embeddings=[query_np.tolist()],
            n_results=n_results
        )

        if not results or not results.get("embeddings"):
            logger.warning("No embeddings retrieved from ChromaDB.")
            return []

        # Prepare for reranking
        candidates = []
        for i, (doc_id, document, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0]
        )):
            # Convert distance to similarity score (ChromaDB returns distances)
            similarity = 1 - distance

            # Update access stats
            self._update_access_stats(doc_id)

            # Calculate final score with frequency and recency
            frequency_score = np.log1p(self.access_stats[doc_id]["count"])
            time_diff = time.time() - self.access_stats[doc_id]["last_access"]
            recency_score = 1.0 / (1.0 + np.log1p(time_diff))

            final_score = (
                    0.6 * similarity +
                    0.2 * frequency_score +
                    0.2 * recency_score
            )

            candidates.append((document, final_score))

            # Update cache
            self._update_cache(doc_id, np.array(results["embeddings"][0][i]), document)

        # Sort by final score and return top k
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    def _check_cache(self, query_embedding: torch.Tensor) -> Optional[List[Tuple[str, float]]]:
        """Check cache for similar embeddings."""
        if not self.cache:
            return None

        cache_results = []
        query_np = query_embedding.cpu().numpy()

        for doc_id, entry in self.cache.items():
            similarity = F.cosine_similarity(
                torch.from_numpy(query_np).unsqueeze(0),
                torch.from_numpy(entry.embedding).unsqueeze(0)
            ).item()

            if similarity >= self.similarity_threshold:
                # Update access stats
                self._update_access_stats(doc_id)
                cache_results.append((entry.document, similarity))

        return sorted(cache_results, key=lambda x: x[1], reverse=True) if cache_results else None

    def _update_cache(self, doc_id: str, embedding: np.ndarray, document: str):
        """Update cache with new entry."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest = min(self.cache.items(), key=lambda x: x[1].last_access)
            del self.cache[oldest[0]]

        self.cache[doc_id] = CacheEntry(
            embedding=embedding,
            document=document,
            access_count=self.access_stats.get(doc_id, {}).get("count", 1),
            last_access=time.time()
        )

    def _update_access_stats(self, doc_id: str):
        """Update access statistics for a document."""
        if doc_id in self.access_stats:
            self.access_stats[doc_id]["count"] += 1
            self.access_stats[doc_id]["last_access"] = time.time()

    def _prune_memory(self):
        """Prune memory using access statistics."""
        if self.collection.count() <= self.max_memory_size * 0.9:
            return

        # Get all documents and their metadata
        results = self.collection.get()

        # Calculate scores for each document
        scores = []
        current_time = time.time()

        for doc_id, document, metadata in zip(results["ids"], results["documents"], results["metadatas"]):
            if doc_id in self.access_stats:
                access_count = self.access_stats[doc_id]["count"]
                last_access = self.access_stats[doc_id]["last_access"]

                recency = current_time - last_access
                score = (
                        0.4 * np.log1p(access_count) +
                        0.6 * (1.0 / (1.0 + np.log1p(recency)))
                )

                scores.append((doc_id, document, score))

        # Sort by score and keep top documents
        scores.sort(key=lambda x: x[2], reverse=True)
        keep_ids = {id for id, _, _ in scores[:self.max_memory_size]}

        # Remove documents not in keep_ids
        remove_ids = [id for id in results["ids"] if id not in keep_ids]
        if remove_ids:
            self.collection.delete(ids=remove_ids)

            # Clean up cache and access stats
            for id in remove_ids:
                self.cache.pop(id, None)
                self.access_stats.pop(id, None)

