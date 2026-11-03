"""Thin wrapper around OpenAI / Azure OpenAI chat that returns validated Pydantic objects.

Structured output strategy: ask for JSON matching a schema, parse it, and
validate it with Pydantic. If validation fails, send the validation error back
to the model once and ask it to fix its output. Nothing downstream ever sees
unvalidated model output.
"""

import json
import logging
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self, client, model: str, provider: str):
        self.client = client
        self.model = model
        self.provider = provider

    def _chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def structured(self, system: str, user: str, schema: type[T], retries: int = 1) -> T:
        schema_hint = json.dumps(schema.model_json_schema())
        messages = [
            {"role": "system", "content": f"{system}\n\nReturn ONLY a JSON object matching this JSON schema:\n{schema_hint}"},
            {"role": "user", "content": user},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            raw = self._chat(messages)
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last_error = e
                log.warning("LLM output failed validation (attempt %d): %s", attempt + 1, e)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON was invalid: {e}. Return corrected JSON only."},
                ]
        raise LLMError(f"Model output failed schema validation: {last_error}")


def get_llm(settings) -> Optional[ChatLLM]:
    """Return a configured chat client, or None for offline mode."""
    provider = settings.resolved_provider()
    if provider == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_endpoint,
        )
        return ChatLLM(client, settings.azure_chat_deployment, "azure")
    if provider == "openai":
        from openai import OpenAI

        return ChatLLM(OpenAI(api_key=settings.openai_api_key), settings.openai_chat_model, "openai")
    return None
