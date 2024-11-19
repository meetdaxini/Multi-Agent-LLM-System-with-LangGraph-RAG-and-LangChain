from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PipelineContext:
    """Holds state and data passed between pipeline steps"""

    documents: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[Any] = None
    queries: Optional[List[str]] = None
    query_embeddings: Optional[Any] = None
    retrieved_documents: Optional[List[List[str]]] = None
    retrieved_metadata: Optional[List[Dict[str, Any]]] = None
    generated_responses: Optional[List[str]] = None


class PipelineStep(ABC):
    """Base class for pipeline steps"""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the pipeline step"""
        pass
