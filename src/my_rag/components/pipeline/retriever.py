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
        pipeline_data.retrieved_documents = [
            [doc_id for doc_id in results["metadatas"][idx]]
            for idx, _ in enumerate(pipeline_data.query_embeddings)
        ]
        pipeline_data.retrieved_metadata = [
            results["metadatas"][idx]
            for idx, _ in enumerate(pipeline_data.query_embeddings)
        ]

        # for idx_in_batch, actual_doc_id in enumerate(pipeline_data.query_embeddings):
        #     retrieved_metadatas = results["metadatas"][idx_in_batch]
        #     retrieved_doc_ids = [metadata["doc_id"] for metadata in retrieved_metadatas]

        #     for doc_id in retrieved_doc_ids:
        #         if doc_id not in unique_retrieved_doc_ids:
        #             unique_retrieved_doc_ids.append(doc_id)
        #     for k in range(self.k):
        #         if actual_doc_id in unique_retrieved_doc_ids[:k]:
        #             correct_counts[k] += 1
        return pipeline_data
