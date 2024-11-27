from typing import List, Optional, Dict, Any
from .base import PipelineStep, PipelineData


class RAGPipeline:
    """Main RAG pipeline that chains together all components"""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(
        self,
        documents: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> PipelineData:
        # Initialize pipeline_data
        pipeline_data = PipelineData(
            documents=documents, document_ids=document_ids, queries=queries
        )

        # Update with any initial pipeline_data
        if initial_context:
            for key, value in initial_context.items():
                setattr(pipeline_data, key, value)

        # Run pipeline steps
        for step in self.steps:
            pipeline_data = step.run(pipeline_data)

        return pipeline_data
