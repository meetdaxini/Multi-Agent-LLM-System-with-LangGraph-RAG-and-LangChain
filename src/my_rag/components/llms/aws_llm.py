import json
import boto3
from botocore.exceptions import ClientError
from typing import Optional, List, Any
from .base import BaseLLM
from ..chat_templates import Message


class AWSBedrockLLM(BaseLLM):
    """LLM implementation using AWS Bedrock with enhanced chat support."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: str = "us-east-1",
        **kwargs,
    ):
        self.model_id = model_id

        try:
            self.client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )

        except Exception as e:
            raise ValueError(f"Failed to initialize AWS Bedrock client: {str(e)}")

    def generate(
        self,
        messages: Any,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate text using messages."""
        body_content = {
            "max_tokens": 1000,
            "temperature": 1.0,
            "top_p": 0.95,
            "system": "\n".join(
                [
                    message["content"]
                    for message in messages
                    if message["role"] == "system"
                ]
            ),
            "messages": [
                message for message in messages if message["role"] != "system"
            ],
            **kwargs,
        }
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body_content),
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]

    def generate_summary(
        self,
        text: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate a summary of the input text."""
        messages = [
            {
                "role": "user",
                "content": f""""Here is a sub-set of documents. Give a detailed summary of the content provided. Documents:{text}""",
            }
        ]

        return self.generate(
            messages=messages,
            max_tokens=1000,
            temperature=1.0,
            top_p=0.95,
            **kwargs,
        )

    def generate_response_with_context(
        self,
        context: str,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate response with context."""
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that provides helpful answers using the given context.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"},
        ]
        return self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def generate_with_template(
        self,
        messages: List[Message],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate response using chat template."""
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        return self.generate(
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def generate_template_response_with_context(
        self,
        context: str,
        query: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Message]] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> str:
        """Generate response with context using chat template."""
        if system_message is None:
            system_message = (
                "You are an AI assistant that provides helpful answers "
                "using the given context. Always base your answers on the "
                "provided context and be precise and concise."
            )
        # Create context-enhanced user message
        user_message = f"Context:\n{context}\n\nQuestion:\n{query}"
        # Create messages list
        messages = [Message(role="system", content=system_message)]
        if chat_history:
            messages.extend(chat_history)
        messages.append(Message(role="user", content=user_message))
        return self.generate_with_template(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def clean_up(self):
        """Clean up resources."""
        if hasattr(self, "client"):
            self.client.close()
