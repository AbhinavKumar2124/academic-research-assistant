"""Semantic Kernel adapter for Ollama's raw local generation endpoint.

Some installed Qwen3 templates force a long thinking preamble. A raw chat prompt
keeps the same model weights and service while avoiding that template behavior.
"""
import httpx
from pydantic import Field
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import AuthorRole, ChatMessageContent


class OllamaRawChatCompletion(ChatCompletionClientBase):
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=180.0, gt=0)

    async def _inner_get_chat_message_contents(self, chat_history, settings):
        pieces = []
        for message in chat_history.messages:
            role = message.role.value
            if role not in {"system", "user", "assistant"}:
                continue
            content = (message.content or "").replace("<|im_start|>", "[im_start]")
            content = content.replace("<|im_end|>", "[im_end]")
            pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        pieces.append("<|im_start|>assistant\n")
        payload = {
            "model": self.ai_model_id,
            "prompt": "".join(pieces),
            "raw": True,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 1500},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url.rstrip("/") + "/api/generate", json=payload)
            response.raise_for_status()
        content = response.json().get("response", "")
        if not content:
            raise ValueError("Ollama returned no answer")
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=content,
                                   ai_model_id=self.ai_model_id)]
