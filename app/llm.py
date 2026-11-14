"""One Semantic Kernel agent interface for either configured provider."""
import json

import httpx
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from app.config import Settings
from app.ollama_service import OllamaRawChatCompletion


class ProviderError(RuntimeError):
    pass


def chat_service(settings: Settings):
    if settings.llm_provider == "ollama":
        return OllamaRawChatCompletion(ai_model_id=settings.ollama_model,
                                       base_url=settings.ollama_base_url)
    if not settings.openrouter_api_key or not settings.openrouter_model:
        raise ProviderError("OpenRouter requires OPENROUTER_API_KEY and OPENROUTER_MODEL")
    client = AsyncOpenAI(api_key=settings.openrouter_api_key,
                         base_url=settings.openrouter_base_url, timeout=120.0)
    return OpenAIChatCompletion(ai_model_id=settings.openrouter_model, async_client=client)


def make_agent(settings: Settings, name: str, instructions: str, plugins: list) -> ChatCompletionAgent:
    return ChatCompletionAgent(
        service=chat_service(settings), name=name, instructions=instructions,
        plugins=plugins,
    )


async def check_connection(settings: Settings) -> str:
    if settings.llm_provider == "openrouter":
        chat_service(settings)  # Validates configuration without leaking credentials.
        return "OpenRouter configured; live response is checked when research runs"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(settings.ollama_base_url.rstrip("/") + "/api/tags")
            response.raise_for_status()
            names = {item["name"] for item in response.json().get("models", [])}
            if settings.ollama_model not in names:
                raise ProviderError(f"Ollama model {settings.ollama_model!r} is not installed")
        history = ChatHistory()
        history.add_user_message('Return {"ok": true} as JSON.')
        result = await chat_service(settings).get_chat_message_content(
            history, PromptExecutionSettings())
        if not result or json.loads(str(result)).get("ok") is not True:
            raise ProviderError("Ollama returned an invalid response")
    except httpx.HTTPError as exc:
        raise ProviderError(f"Ollama is unavailable: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc
    return f"Ollama responded with {settings.ollama_model}"
