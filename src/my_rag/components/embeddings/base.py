from abc import ABC, abstractmethod
from typing import List, Any

class BaseEmbedding(ABC):
    """
    Abstract base class for embedding models.
    """

    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> Any:
        """
        Embeds a list of texts.

        Args:
            texts (List[str]): The texts to embed.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The embeddings of the input texts.
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        pass
