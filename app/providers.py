"""LLM provider adapters.

Every engine talks OpenAI chat-completions to the internal proxy; this module
maps that request onto the configured provider's native wire format.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from . import oauth

Messages = list[dict[str, Any]]

KINDS = ("openai_compatible", "openai", "anthropic", "xai", "gemini")

# UI presets. `auth` lists the allowed methods; the first is the default.
PRESETS: dict[str, dict[str, Any]] = {
    "openai_compatible": {
        "label": "로컬 / OpenAI 호환",
        "auth": ["api_key", "none"],
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "deepseek-v4-flash-0731",
        "extra_body": {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}, "max_tokens": 16384},
    },
    "openai": {"label": "OpenAI", "auth": ["oauth", "api_key"], "model": "gpt-5.5", "extra_body": {}},
    "anthropic": {"label": "Anthropic", "auth": ["api_key"], "model": "claude-sonnet-5", "extra_body": {}},
    "xai": {"label": "xAI", "auth": ["oauth", "api_key"], "model": "grok-4.6", "extra_body": {}},
    "gemini": {"label": "Google Gemini", "auth": ["api_key"], "model": "gemini-3-pro", "extra_body": {}},
}

OPENAI_API = "https://api.openai.com/v1"
ANTHROPIC_API = "https://api.anthropic.com/v1"
XAI_API = "https://api.x.ai/v1"
GEMINI_OPENAI_API = "https://generativelanguage.googleapis.com/v1beta/openai"
CODEX_RESPONSES = "https://chatgpt.com/backend-api/codex/responses"

TIMEOUT = httpx.Timeout(600.0, connect=20.0)


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class ChatRequest:
    messages: Messages
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    id: str
    name: str
    kind: str
    auth: str
    base_url: str | None
    model: str
    extra_body: dict[str, Any]
    vision: bool
    secret: dict[str, Any] | None


SecretSaver = Callable[[dict[str, Any]], Awaitable[None]]


# --------------------------------------------------------------------------- helpers


def strip_code_fence(text: str) -> str:
    """Remove a single surrounding ```json fence that chat models like to add."""
    match = re.fullmatch(r"\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```\s*", text, flags=re.S)
    return match.group(1) if match else text


def text_only(messages: Messages) -> Messages:
    """Drop image parts for providers/models configured without vision."""
    result: Messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = [part for part in content if part.get("type") == "text"]
            content = "\n".join(part.get("text", "") for part in parts)
        result.append({**message, "content": content})
    return result


def _parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _data_url(url: str) -> tuple[str, str]:
    match = re.match(r"data:([^;]+);base64,(.*)", url, flags=re.S)
    if not match:
        raise ProviderError("이미지는 data URL(base64) 형식만 지원합니다.", 400)
    return match.group(1), match.group(2)


def _schema_instruction(response_format: dict[str, Any] | None) -> str | None:
    if not response_format or response_format.get("type") not in ("json_schema", "json_object"):
        return None
    schema = (response_format.get("json_schema") or {}).get("schema")
    base = "Respond with a single JSON object only, without markdown fences or commentary."
    return f"{base}\nThe JSON must match this JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}" if schema else base


def _split_system(messages: Messages) -> tuple[str, Messages]:
    system: list[str] = []
    rest: Messages = []
    for message in messages:
        if message.get("role") in ("system", "developer"):
            system.extend(part.get("text", "") for part in _parts(message.get("content")) if part.get("type") == "text")
        else:
            rest.append(message)
    return "\n\n".join(filter(None, system)), rest


# --------------------------------------------------------------------------- OpenAI chat-completions family


async def _chat_completions(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    request: ChatRequest,
    extra_body: dict[str, Any],
    *,
    max_tokens_field: str = "max_tokens",
    send_sampling: bool = True,
) -> str:
    body: dict[str, Any] = {"model": model, "messages": request.messages}
    if request.max_tokens:
        body[max_tokens_field] = request.max_tokens
    if send_sampling and request.temperature is not None:
        body["temperature"] = request.temperature
    if request.response_format:
        body["response_format"] = request.response_format
    body.update(extra_body)
    if max_tokens_field != "max_tokens" and "max_tokens" in body:
        body[max_tokens_field] = body.pop("max_tokens")
    response = await client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        raise ProviderError(f"{url} → HTTP {response.status_code}: {response.text[:800]}", response.status_code)
    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    if not content:
        reason = choice.get("finish_reason")
        raise ProviderError(f"모델이 빈 응답을 반환했습니다 (finish_reason={reason}). max_tokens 또는 추론 설정을 확인하세요.")
    return content


# --------------------------------------------------------------------------- Anthropic Messages


def to_anthropic(messages: Messages) -> tuple[str, list[dict[str, Any]]]:
    system, rest = _split_system(messages)
    converted: list[dict[str, Any]] = []
    for message in rest:
        blocks: list[dict[str, Any]] = []
        for part in _parts(message.get("content")):
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                media_type, data = _data_url(part["image_url"]["url"])
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        role = "assistant" if message.get("role") == "assistant" else "user"
        converted.append({"role": role, "content": blocks})
    return system, converted


async def _anthropic(client: httpx.AsyncClient, config: ProviderConfig, request: ChatRequest) -> str:
    api_key = (config.secret or {}).get("api_key")
    if not api_key:
        raise ProviderError("Anthropic API 키가 설정되지 않았습니다.", 401)
    system, messages = to_anthropic(request.messages)
    instruction = _schema_instruction(request.response_format)
    if instruction:
        system = f"{system}\n\n{instruction}" if system else instruction
    body: dict[str, Any] = {"model": config.model, "max_tokens": request.max_tokens or 8192, "messages": messages}
    if system:
        body["system"] = system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    body.update(config.extra_body)
    response = await client.post(
        f"{config.base_url or ANTHROPIC_API}/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json=body,
    )
    if response.status_code >= 400:
        raise ProviderError(f"Anthropic HTTP {response.status_code}: {response.text[:800]}", response.status_code)
    blocks = response.json().get("content") or []
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
    if not text:
        raise ProviderError("Anthropic 응답에 텍스트가 없습니다.")
    return strip_code_fence(text) if instruction else text


# --------------------------------------------------------------------------- OpenAI Responses (ChatGPT/Codex OAuth)


def to_responses_input(messages: Messages) -> tuple[str, list[dict[str, Any]]]:
    system, rest = _split_system(messages)
    items: list[dict[str, Any]] = []
    for message in rest:
        assistant = message.get("role") == "assistant"
        content: list[dict[str, Any]] = []
        for part in _parts(message.get("content")):
            if part.get("type") == "text":
                content.append({"type": "output_text" if assistant else "input_text", "text": part.get("text", "")})
            elif part.get("type") == "image_url" and not assistant:
                content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        items.append({"type": "message", "role": "assistant" if assistant else "user", "content": content})
    return system, items


def _responses_text_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        spec = response_format.get("json_schema") or {}
        return {
            "type": "json_schema",
            "name": spec.get("name", "response"),
            "schema": spec.get("schema", {}),
            "strict": bool(spec.get("strict", False)),
        }
    if response_format.get("type") == "json_object":
        return {"type": "json_object"}
    return None


async def _codex(
    client: httpx.AsyncClient, config: ProviderConfig, request: ChatRequest, save_secret: SecretSaver
) -> str:
    tokens = await oauth.codex_valid_tokens(client, config.secret, save_secret)
    system, items = to_responses_input(request.messages)
    body: dict[str, Any] = {
        "model": config.model,
        "instructions": system or "You are a professional manga translator.",
        "input": items,
        "stream": True,
        "store": False,
    }
    text_format = _responses_text_format(request.response_format)
    if text_format:
        body["text"] = {"format": text_format}
    body.update(config.extra_body)
    headers = {
        "Authorization": f"Bearer {tokens['access_token']}",
        "chatgpt-account-id": tokens.get("account_id", ""),
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "session_id": str(uuid.uuid4()),
        "Accept": "text/event-stream",
    }
    chunks: list[str] = []
    completed_text: str | None = None
    async with client.stream("POST", CODEX_RESPONSES, headers=headers, json=body) as response:
        if response.status_code >= 400:
            detail = (await response.aread()).decode(errors="replace")[:800]
            raise ProviderError(f"ChatGPT(Codex) HTTP {response.status_code}: {detail}", response.status_code)
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            event = json.loads(payload)
            kind = event.get("type")
            if kind == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif kind == "response.completed":
                completed_text = _responses_output_text(event.get("response") or {})
            elif kind in ("response.failed", "error"):
                raise ProviderError(f"ChatGPT(Codex) 응답 실패: {json.dumps(event, ensure_ascii=False)[:800]}")
    text = completed_text or "".join(chunks)
    if not text:
        raise ProviderError("ChatGPT(Codex) 응답에 텍스트가 없습니다.")
    return text


def _responses_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    return "".join(texts)


# --------------------------------------------------------------------------- dispatch


async def chat(config: ProviderConfig, request: ChatRequest, save_secret: SecretSaver) -> str:
    if not config.vision:
        request = ChatRequest(text_only(request.messages), request.max_tokens, request.temperature, request.response_format)
    secret = config.secret or {}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if config.kind == "openai_compatible":
            if not config.base_url:
                raise ProviderError("OpenAI 호환 제공자에는 base URL이 필요합니다.", 400)
            headers = {"Authorization": f"Bearer {secret['api_key']}"} if secret.get("api_key") else {}
            return await _chat_completions(
                client, f"{config.base_url.rstrip('/')}/chat/completions", headers, config.model, request, config.extra_body
            )
        if config.kind == "openai":
            if config.auth == "oauth":
                return await _codex(client, config, request, save_secret)
            return await _chat_completions(
                client,
                f"{(config.base_url or OPENAI_API).rstrip('/')}/chat/completions",
                _bearer(secret.get("api_key"), "OpenAI"),
                config.model,
                request,
                config.extra_body,
                max_tokens_field="max_completion_tokens",
            )
        if config.kind == "anthropic":
            return await _anthropic(client, config, request)
        if config.kind == "xai":
            if config.auth == "oauth":
                tokens = await oauth.xai_valid_tokens(client, config.secret, save_secret)
                headers = {"Authorization": f"Bearer {tokens['access_token']}"}
            else:
                headers = _bearer(secret.get("api_key"), "xAI")
            return await _chat_completions(
                client, f"{(config.base_url or XAI_API).rstrip('/')}/chat/completions", headers, config.model, request, config.extra_body
            )
        if config.kind == "gemini":
            return await _chat_completions(
                client,
                f"{(config.base_url or GEMINI_OPENAI_API).rstrip('/')}/chat/completions",
                _bearer(secret.get("api_key"), "Gemini"),
                config.model,
                request,
                config.extra_body,
            )
    raise ProviderError(f"알 수 없는 제공자 종류: {config.kind}", 400)


def _bearer(api_key: str | None, label: str) -> dict[str, str]:
    if not api_key:
        raise ProviderError(f"{label} API 키가 설정되지 않았습니다.", 401)
    return {"Authorization": f"Bearer {api_key}"}


def completion_envelope(model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def decode_data_url_bytes(url: str) -> bytes:
    return base64.b64decode(_data_url(url)[1])
