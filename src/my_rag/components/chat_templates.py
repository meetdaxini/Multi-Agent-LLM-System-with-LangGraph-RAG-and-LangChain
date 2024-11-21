from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str


class ChatTemplateManager:
    """Manages chat templates for different models"""

    @staticmethod
    def format_messages(messages: List[Message]) -> str:
        """Formats messages according to the model's template"""
        return [asdict(message) for message in messages]

    @staticmethod
    def create_prompt(
        system_message: str,
        user_message: str,
        chat_history: Optional[List[Message]] = None,
    ) -> str:
        """Creates a formatted prompt with optional chat history"""
        messages = []

        messages.append(Message(role="system", content=system_message))

        if chat_history:
            messages.extend(chat_history)

        messages.append(Message(role="user", content=user_message))

        return ChatTemplateManager.format_messages(messages)
