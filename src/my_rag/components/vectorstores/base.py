from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """

    @abstractmethod
    def add_embeddings(
        self,
        embeddings: Any,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Adds embeddings to the vector store.

        Args:
            embeddings (Any): The embeddings to add.
            documents (List[str]): The original text documents.
            metadatas (Optional[List[dict]]): Optional metadata for each embedding.
            ids (Optional[List[str]]): Optional IDs for each embedding.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embeddings: Any,
        k: int = 5,
        filter_dict: Optional[Dict] = None,
        include: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Searches for the top k most similar embeddings.

        Args:
            query_embeddings (Any): The embedding of the query.
            k (int): The number of top results to return.
            filter_dict (Optional[Dict]): Optional filters for the search.
            include (Optional[List[str]]): What to include in results.

        Returns:
            List[dict]: A list of results with embeddings and metadata.
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict[str, Any]: Collection statistics
        """
        pass
