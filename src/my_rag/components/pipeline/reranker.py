from my_rag.components.reranking.base import BaseReranker
from .base import PipelineStep, PipelineData


class RerankerStep(PipelineStep):
    """Pipeline step for reranking retrieved documents"""

    def __init__(self, reranker: BaseReranker, k: int = 5):
        self.reranker = reranker
        self.k = k

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        if not pipeline_data.queries or not pipeline_data.retrieved_documents:
            return pipeline_data

        reranked_documents = []
        reranked_metadata = []

        for query, docs, metadata in zip(
            pipeline_data.queries,
            pipeline_data.retrieved_documents,
            pipeline_data.retrieved_metadata,
        ):
            # Rerank documents
            reranked_results = self.reranker.rerank(
                query=query, documents=docs, k=min(self.k, len(docs))
            )

            # Reorder documents and metadata based on new ranking
            ordered_docs = []
            ordered_metadata = []
            for result in reranked_results:
                ordered_docs.append(result["content"])
                ordered_metadata.append(metadata[result["result_index"]])

            reranked_documents.append(ordered_docs)
            reranked_metadata.append(ordered_metadata)

        pipeline_data.retrieved_documents = reranked_documents
        pipeline_data.retrieved_metadata = reranked_metadata
        return pipeline_data
