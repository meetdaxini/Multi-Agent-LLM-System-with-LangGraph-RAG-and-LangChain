from typing import Optional, List, Dict, Any
from .base import PipelineStep, PipelineData
from ..llms.base import BaseLLM
from ..chat_templates import Message


class Generator(PipelineStep):
    """Generates responses using an LLM and Context with chat template support"""

    def __init__(
        self,
        llm: BaseLLM,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Message]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Generator.

        Args:
            llm: The language model to use for generation
            system_message: System message for the chat
            chat_history: Optional chat history
            generation_config: Configuration for text generation
        """
        self.llm = llm
        self.system_message = system_message or (
            "You are an AI assistant that provides accurate and helpful answers "
            "based on the given context. Your responses should be:"
            "\n1. Focused on the provided context"
            "\n2. Clear and concise"
            "\n3. Accurate and relevant to the question"
            "\n4. Based only on the information given"
        )
        self.chat_history = chat_history or []
        self.generation_config = generation_config or {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

    def _prepare_context_message(self, context: List[str], query: str) -> str:
        """Prepares the context and query into a formatted message"""
        context_str = "\n".join(context)
        return (
            f"Please answer the following question using the provided documents as context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}"
        )

    def run(self, pipeline_data: PipelineData) -> PipelineData:
        """
        Runs the generation step.

        Args:
            pipeline_data: Pipeline data containing queries and retrieved documents

        Returns:
            Updated pipeline data with generated responses
        """
        if not pipeline_data.queries or not pipeline_data.retrieved_documents:
            raise ValueError("Queries and retrieved documents must be provided")

        responses = []
        for query, docs in zip(
            pipeline_data.queries, pipeline_data.retrieved_documents
        ):
            # Create the context message
            context_message = self._prepare_context_message(docs, query)

            # Generate response using template
            response = self.llm.generate_template_response_with_context(
                context=context_message,
                query=query,
                system_message=self.system_message,
                chat_history=self.chat_history,
                **self.generation_config,
            )
            responses.append(response)

        pipeline_data.generated_responses = responses
        return pipeline_data

    @classmethod
    def from_config(cls, config: Dict[str, Any], llm: BaseLLM) -> "Generator":
        """
        Creates a Generator instance from a configuration dictionary.

        Args:
            config: Configuration dictionary
            llm: Language model instance

        Returns:
            Configured Generator instance
        """
        return cls(
            llm=llm,
            system_message=config.get("system_message"),
            chat_history=(
                [Message(**msg) for msg in config.get("chat_history", [])]
                if config.get("chat_history")
                else None
            ),
            generation_config=config.get("generation_config"),
        )
