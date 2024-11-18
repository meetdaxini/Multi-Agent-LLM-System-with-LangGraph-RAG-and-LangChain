from typing import List, Dict, Any, Optional
import chromadb
from enum import Enum
from .base import BaseVectorStore


class CollectionMode(Enum):
    """Enum defining modes for handling existing collections"""

    FAIL_IF_EXISTS = "fail_if_exists"
    DROP_IF_EXISTS = "drop_if_exists"
    CREATE_IF_NOT_EXISTS = "create_if_not_exists"


class ChromaVectorStore(BaseVectorStore):
    """Vector store implementation using ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        distance_metric: str = "cosine",
        mode: CollectionMode = CollectionMode.FAIL_IF_EXISTS,
    ):
        """
        Initialize the ChromaDB vector store.

        Args:
            collection_name (str): Name of the collection
            host (str): ChromaDB host
            port (int): ChromaDB port
            distance_metric (str): Distance metric for similarity search
            mode (CollectionMode): Mode for handling existing collections

        Raises:
            ValueError: If collection exists and mode is FAIL_IF_EXISTS
        """
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
        )

        # Check if collection exists
        existing_collections = self.client.list_collections()
        collection_exists = any(
            col.name == collection_name for col in existing_collections
        )

        if collection_exists:
            if mode == CollectionMode.FAIL_IF_EXISTS:
                raise ValueError(f"Collection '{collection_name}' already exists")

            elif mode == CollectionMode.DROP_IF_EXISTS:
                self.client.delete_collection(name=collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name, metadata={"hnsw:space": distance_metric}
                )

            elif mode == CollectionMode.CREATE_IF_NOT_EXISTS:
                self.collection = self.client.get_collection(
                    name=collection_name, embedding_function=None
                )

        else:
            # Collection doesn't exist, create new
            self.collection = self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": distance_metric}
            )

    def add_embeddings(
        self,
        embeddings: Any,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add embeddings and their metadata to the store.

        Args:
            embeddings (Any): Document embeddings
            documents (List[str]): Original text documents
            metadatas (Optional[List[dict]]): Metadata for each document
            ids (Optional[List[str]]): Optional custom IDs for the embeddings
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"doc_id": doc_id} for doc_id in ids]

        self.collection.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
        )

    def search(
        self,
        query_embeddings: Any,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Search for similar vectors.

        Args:
            query_embeddings (Any): Query embeddings
            k (int): Number of results to return
            filter_dict (Optional[Dict]): Filters for the search
            include (Optional[List[str]]): What to include in results

        Returns:
            List[dict]: Search results
        """
        if include is None:
            include = ["metadatas", "documents", "distances"]

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            where=filter_dict,
            include=include,
        )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics
        """
        return {
            "count": self.collection.count(),
            "name": self.collection_name,
            "metadata": self.collection.metadata,
        }

    def clean_up(self):
        """Clean up resources."""
        self.delete_collection()

    def delete_collection(self):
        """Delete the current collection."""
        self.client.delete_collection(name=self.collection_name)
