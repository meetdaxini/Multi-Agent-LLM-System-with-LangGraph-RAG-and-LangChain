from typing import List, Optional, Dict, Any
from .base import PipelineStep, PipelineContext


class RAGPipeline:
    """Main RAG pipeline that chains together all components"""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(
        self,
        documents: List[str],
        document_ids: List[str],
        queries: List[str],
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineContext:
        # Initialize context
        context = PipelineContext(
            documents=documents, document_ids=document_ids, queries=queries
        )

        # Update with any initial context
        if initial_context:
            for key, value in initial_context.items():
                setattr(context, key, value)

        # Run pipeline steps
        for step in self.steps:
            context = step.run(context)

        return context
