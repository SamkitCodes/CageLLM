from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def _format_messages_history(self, messages) -> str:
        if isinstance(messages, str):
            return messages
        """Format the conversation history for the LLM."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|system|>{content}"
            elif role == "user":
                formatted += f"<|user|>{content}"
            elif role == "assistant":
                formatted += f"<|assistant|>{content}"
        return formatted + "<|assistant|>"
    
    def _format_response(self, response) -> str:
        """Format the response from the LLM."""
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()
        return response