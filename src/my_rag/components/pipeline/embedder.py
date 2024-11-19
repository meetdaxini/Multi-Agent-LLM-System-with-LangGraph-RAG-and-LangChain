from typing import Optional
from .base import PipelineStep, PipelineContext
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

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.documents:
            raise ValueError("Documents must be provided for embedding")

        context.embeddings = self.model.embed(
            context.documents, batch_size=self.batch_size, instruction=self.instruction
        )
        return context


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

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.queries:
            raise ValueError("Queries must be provided for embedding")

        context.query_embeddings = self.model.embed(
            context.queries, batch_size=self.batch_size, instruction=self.instruction
        )
        return context
