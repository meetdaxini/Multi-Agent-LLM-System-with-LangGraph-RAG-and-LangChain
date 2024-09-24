from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract base class for language models (LLMs).
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates text based on the input prompt.

        Args:
            prompt (str): The input prompt for the language model.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Cleans up resources to free memory.
        """
        pass
