from typing import Optional
from .base import PipelineStep, PipelineData
from ..embeddings.base import BaseEmbedding


class DocumentEmbedder(PipelineStep):
    """Embeds documents using the provided embedding model"""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        batch_size: int = 32,
        instruction: Optional[str] = None,
    ):
        self.model = embedding_model
        self.batch_size = batch_size
        self.instruction = instruction

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        if not pipeline_data.documents:
            return pipeline_data

        pipeline_data.embeddings = self.model.embed(
            pipeline_data.documents,
            batch_size=self.batch_size,
            instruction=self.instruction,
        )
        return pipeline_data


class QueryEmbedder(PipelineStep):
    """Embeds queries using the provided embedding model"""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        batch_size: int = 32,
        instruction: Optional[str] = None,
    ):
        self.model = embedding_model
        self.batch_size = batch_size
        self.instruction = instruction

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        if not pipeline_data.queries:
            return pipeline_data

        pipeline_data.query_embeddings = self.model.embed(
            pipeline_data.queries,
            batch_size=self.batch_size,
            instruction=self.instruction,
        )
        return pipeline_data
