import json

import httpx
import pytest
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory

from app.config import Settings
from app.llm import ProviderError, chat_service, make_agent
from app.ollama_service import OllamaRawChatCompletion
from app.tools.query_tools import QueryTools


def test_semantic_kernel_agent_has_tools(settings):
    agent = make_agent(settings, "QueryAgent", "Generate queries", [QueryTools()])
    assert isinstance(agent, ChatCompletionAgent)
    assert {"extract_keywords", "validate_and_normalize_queries"}.issubset(
        set(agent.kernel.plugins["QueryTools"].functions))
    assert isinstance(chat_service(settings), OllamaRawChatCompletion)


def test_openrouter_configuration_is_optional(settings):
    assert isinstance(chat_service(settings), OllamaRawChatCompletion)
    with pytest.raises(ProviderError):
        chat_service(Settings(llm_provider="openrouter", openrouter_api_key="",
                              openrouter_model="example/model"))
    service = chat_service(Settings(llm_provider="openrouter", openrouter_api_key="test-key",
                                    openrouter_model="example/model"))
    assert service.ai_model_id == "example/model"


@pytest.mark.asyncio
async def test_raw_ollama_adapter(monkeypatch, settings):
    captured = {}

    class Client:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            captured.update(json)
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  json={"response": '{"queries":["hospital learning"]}'})

    monkeypatch.setattr("app.ollama_service.httpx.AsyncClient", Client)
    history = ChatHistory()
    history.add_user_message("Generate hospital queries")
    result = await chat_service(settings).get_chat_message_content(history,
                                                                     chat_service(settings).instantiate_prompt_execution_settings())
    assert json.loads(str(result))["queries"] == ["hospital learning"]
    assert captured["model"] == "qwen3:4b"
    assert captured["raw"] is True and captured["format"] == "json"
    assert "<|im_start|>user" in captured["prompt"]
