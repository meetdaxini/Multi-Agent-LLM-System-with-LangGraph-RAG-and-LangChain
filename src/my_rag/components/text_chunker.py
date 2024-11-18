from typing import List, Dict, Any, Optional, Union, Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass, asdict


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""

    doc_id: str
    chunk_index: int
    total_chunks: int
    source_type: str
    # additional_metadata: Optional[Dict[str, Any]] = {}


class TextChunker:
    """A class to handle text chunking operations."""

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 250,
        length_function: callable = len,
        separators: Optional[List[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
    ):
        """
        Initialize the TextChunker.

        Args:
            chunk_size (int): The size of each chunk
            chunk_overlap (int): The overlap between chunks
            length_function (callable): Function to measure text length
            separators (Optional[List[str]]): Custom separators for text splitting
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            keep_separator=keep_separator,
            is_separator_regex=is_separator_regex,
        )

    def create_chunks(
        self,
        text: str,
        doc_id: str,
        source_type: str = "text",
        # additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[str], List[ChunkMetadata]]:
        """
        Split text into chunks with metadata.

        Args:
            text (str): Text to split
            doc_id (str): Document identifier
            source_type (str): Type of source document
            additional_metadata (Optional[Dict[str, Any]]): Additional metadata

        Returns:
            tuple[List[str], List[ChunkMetadata]]: Chunks and their metadata
        """
        chunks = self.text_splitter.split_text(text)
        total_chunks = len(chunks)

        metadata = [
            asdict(
                ChunkMetadata(
                    doc_id=doc_id,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    source_type=source_type,
                    # additional_metadata=additional_metadata,
                )
            )
            for i in range(total_chunks)
        ]

        return chunks, metadata

    def create_chunks_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        source_type: str = "text",
        # additional_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[str], List[ChunkMetadata]]:
        """
        Process multiple texts into chunks with metadata.

        Args:
            texts (List[str]): List of texts to split
            doc_ids (List[str]): List of document identifiers
            source_type (str): Type of source documents
            # additional_metadata (Optional[List[Dict[str, Any]]]): Additional metadata for each text

        Returns:
            tuple[List[str], List[ChunkMetadata]]: All chunks and their metadata
        """
        all_chunks = []
        all_metadata = []

        for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
            # metadata = additional_metadata[i] if additional_metadata else None
            chunks, chunk_metadata = self.create_chunks(
                text,
                doc_id,
                source_type,  # metadata
            )
            all_chunks.extend(chunks)
            all_metadata.extend(chunk_metadata)

        return all_chunks, all_metadata
