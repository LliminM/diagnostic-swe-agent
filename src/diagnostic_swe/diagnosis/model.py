"""A fresh model conversation for each diagnostic decision."""

import json
from typing import Any, Protocol

class DiagnosticModel(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class LiteLLMDiagnosticModel:
    """Use a separate LiteLLM call without the coding agent's messages or tools."""

    def __init__(self, model_name: str, **model_kwargs: Any):
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if {"messages", "tools", "stream"} & model_kwargs.keys():
            raise ValueError("messages, tools and stream are controlled by the diagnostic model")
        self.model_name = model_name
        self.model_kwargs = model_kwargs

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        import litellm

        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **self.model_kwargs,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("diagnostic model returned no JSON text")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("diagnostic model must return one JSON object")
        return data
