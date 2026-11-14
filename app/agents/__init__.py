"""Five Semantic Kernel agents with validated Python tool boundaries."""
import json
import re

from pydantic import BaseModel


class AgentOutputError(ValueError):
    pass


async def ask_json(agent, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Limited retries for small local models that occasionally wrap JSON in prose."""
    instruction = (prompt + "\nReturn only a JSON object matching this schema: "
                   + json.dumps(schema.model_json_schema()) + "\nDo not use Markdown fences.")
    for attempt in range(2):
        response = await agent.get_response(messages=instruction)
        raw = str(response.content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return schema.model_validate(json.loads(match.group()))
            except (ValueError, json.JSONDecodeError):
                pass
        instruction = (prompt + "\nYour previous output was invalid. Return exactly one JSON object "
                       f"matching this schema: {json.dumps(schema.model_json_schema())}")
    raise AgentOutputError(f"{agent.name} did not return valid {schema.__name__} JSON")
