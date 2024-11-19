from typing import Optional
from .base import PipelineStep, PipelineContext
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

    def run(self, context: PipelineContext) -> PipelineContext:
        # Initialize vector store with document embeddings if not already done
        self.vector_store.add_embeddings(
            embeddings=context.embeddings,
            documents=context.documents,
            metadatas=context.metadata,
        )

        # Get results for each query
        results = self.vector_store.search(
            query_embeddings=context.query_embeddings,
            k=self.k,
            filter_dict=self.filter_fn() if self.filter_fn else None,
        )
        context.retrieved_documents = [
            [doc_id for doc_id in results["metadatas"][idx]]
            for idx, _ in enumerate(context.query_embeddings)
        ]
        context.retrieved_metadata = [
            results["metadatas"][idx] for idx, _ in enumerate(context.query_embeddings)
        ]

        # for idx_in_batch, actual_doc_id in enumerate(context.query_embeddings):
        #     retrieved_metadatas = results["metadatas"][idx_in_batch]
        #     retrieved_doc_ids = [metadata["doc_id"] for metadata in retrieved_metadatas]

        #     for doc_id in retrieved_doc_ids:
        #         if doc_id not in unique_retrieved_doc_ids:
        #             unique_retrieved_doc_ids.append(doc_id)
        #     for k in range(self.k):
        #         if actual_doc_id in unique_retrieved_doc_ids[:k]:
        #             correct_counts[k] += 1
        return context
