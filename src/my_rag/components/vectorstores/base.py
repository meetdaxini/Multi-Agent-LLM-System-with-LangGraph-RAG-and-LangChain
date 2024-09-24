from abc import ABC, abstractmethod
from typing import Any, List, Optional

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    """

    @abstractmethod
    def add_embeddings(self, embeddings: Any, metadata: Optional[List[dict]] = None):
        """
        Adds embeddings to the vector store.

        Args:
            embeddings (Any): The embeddings to add.
            metadata (Optional[List[dict]]): Optional metadata for each embedding.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: Any, k: int) -> List[dict]:
        """
        Searches for the top k most similar embeddings.

        Args:
            query_embedding (Any): The embedding of the query.
            k (int): The number of top results to return.

        Returns:
            List[dict]: A list of results with embeddings and metadata.
        """
        pass

    # @abstractmethod
    # def save(self, path: str):
    #     """
    #     Saves the vector store to a file.

    #     Args:
    #         path (str): The file path to save the vector store.
    #     """
    #     pass

    @abstractmethod
    def load(self, path: str):
        """
        Loads the vector store from a file.

        Args:
            path (str): The file path to load the vector store from.
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        pass
