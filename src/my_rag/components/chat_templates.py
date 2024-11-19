from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Message:
    role: str
    content: str
    name: Optional[str] = None


class ChatTemplateManager:
    """Manages chat templates for different models"""

    @staticmethod
    def get_template_config(model_name: str) -> Dict[str, Any]:
        """Returns template configuration for specific models"""
        templates = {
            "default": {
                "system_template": "System: {system_message}\n",
                "user_template": "User: {message}\n",
                "assistant_template": "Assistant: {message}\n",
                "system_message_required": False,
                "response_prefix": "Assistant:",
            },
        }
        return templates.get(model_name, templates["default"])

    @staticmethod
    def format_messages(
        messages: List[Message], model_name: str, add_generation_prompt: bool = True
    ) -> str:
        """Formats messages according to the model's template"""
        template_config = ChatTemplateManager.get_template_config(model_name)
        formatted_chat = ""

        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages and template_config["system_message_required"]:
            formatted_chat += template_config["system_template"].format(
                system_message=system_messages[0].content
            )

        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                formatted_chat += template_config["user_template"].format(
                    message=msg.content
                )
            elif msg.role == "assistant":
                formatted_chat += template_config["assistant_template"].format(
                    message=msg.content
                )

        if add_generation_prompt:
            formatted_chat += template_config["response_prefix"]

        return formatted_chat

    @staticmethod
    def create_prompt_from_template(
        system_message: str,
        user_message: str,
        chat_history: Optional[List[Message]] = None,
        model_name: str = "default",
    ) -> str:
        """Creates a formatted prompt with optional chat history"""
        messages = []

        messages.append(Message(role="system", content=system_message))

        if chat_history:
            messages.extend(chat_history)

        messages.append(Message(role="user", content=user_message))

        return ChatTemplateManager.format_messages(messages, model_name)
