"""
Class for recursively chunking text using RecursiveCharacterTextSplitter.

This module provides:

— TextSplitter
"""

from typing import Union, List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    """Class for recursively chunking text, prioritizing '\n\n', '\n', and other delimiters.

    Attributes:
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Overlap between text chunks.
    """

    def __init__(
        self,
        chunk_size: Union[int, str] = 2000,
        chunk_overlap: Union[int, str] = 400
    ):
        """Initialize TextSplitter with chunk_size and chunk_overlap."""
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str) -> List[str]:
        """Split the input text into chunks based on the provided settings.

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        return self.text_splitter.split_text(text)

    def set_chunk_size(self, chunk_size: int):
        """Set a new chunk size for text splitting.

        Args:
            chunk_size (int): The new chunk size.
        """
        self.chunk_size = chunk_size
        self.text_splitter.chunk_size = chunk_size

    def set_chunk_overlap(self, chunk_overlap: int):
        """Set a new chunk overlap for text splitting.

        Args:
            chunk_overlap (int): The new chunk overlap.
        """
        self.chunk_overlap = chunk_overlap
        self.text_splitter.chunk_overlap = chunk_overlap

# Example usage:
if __name__ == "__main__":
    # Example text to be split
    sample_text = (
        "This is an example text that needs to be split into smaller chunks.\n\n"
        "It contains various paragraphs, lines, and other characters.\n"
        "The purpose is to chunk this large text efficiently using the RecursiveCharacterTextSplitter."
    )

    # Initialize the TextSplitter with default chunk_size and chunk_overlap
    splitter = TextSplitter(chunk_size=100, chunk_overlap=10)

    # Split the text and print each chunk
    chunks = splitter.split_text(sample_text)

    print("Text has been split into the following chunks:")
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {chunk}\n{'-'*40}")
