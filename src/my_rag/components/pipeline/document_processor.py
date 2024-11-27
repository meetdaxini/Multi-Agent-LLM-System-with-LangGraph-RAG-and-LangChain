from typing import Any, Dict, Optional, Callable
from .base import PipelineStep, PipelineData
from ..text_chunker import TextChunker


class DocumentProcessor(PipelineStep):
    """Processes documents and prepares them for embedding"""

    def __init__(
        self,
        doc_processor_fn: Optional[Callable[[str], str]] = None,
        metadata_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 250,
    ):
        self.doc_processor_fn = doc_processor_fn or (lambda x: x)
        self.metadata_fn = metadata_fn or (lambda doc, id: {"doc_id": id})
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        if not pipeline_data.documents or not pipeline_data.document_ids:
            return pipeline_data

        processed_chunks = []
        processed_metadata = []

        for doc, doc_id in zip(pipeline_data.documents, pipeline_data.document_ids):
            # Process document
            processed_doc = self.doc_processor_fn(doc)

            # Create chunks
            chunks, chunk_metadata = self.chunker.create_chunks(
                processed_doc, doc_id, source_type="text"
            )

            # Add custom metadata
            for chunk, base_metadata in zip(chunks, chunk_metadata):
                base_metadata.update(self.metadata_fn(chunk, doc_id))
                processed_chunks.append(chunk)
                processed_metadata.append(base_metadata)

        pipeline_data.documents = processed_chunks
        pipeline_data.metadata = processed_metadata
        return pipeline_data
