from typing import Optional
from .base import PipelineStep, PipelineData
from ..vectorstores.base import BaseVectorStore


class Retriever(PipelineStep):
    """Retrieves relevant documents for queries"""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 5,
        filter_fn: Optional[callable] = None,
    ):
        self.vector_store = vector_store
        self.k = k
        self.filter_fn = filter_fn

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        # Initialize vector store with document embeddings if not already done
        self.vector_store.add_embeddings(
            embeddings=pipeline_data.embeddings,
            documents=pipeline_data.documents,
            metadatas=pipeline_data.metadata,
        )

        # Get results for each query
        results = self.vector_store.search(
            query_embeddings=pipeline_data.query_embeddings,
            k=self.k,
            filter_dict=self.filter_fn() if self.filter_fn else None,
        )
        pipeline_data.retrieved_documents = results["documents"]
        pipeline_data.retrieved_metadata = results["metadatas"]

        return pipeline_data
