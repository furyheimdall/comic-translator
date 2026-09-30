import pytest

from app.providers import ProviderConfig, ProviderError, reasoning_levels, with_overrides


def provider(kind, auth="api_key", extra_body=None):
    return ProviderConfig(
        id="p", name="p", kind=kind, auth=auth, base_url=None, model="base-model",
        extra_body=extra_body if extra_body is not None else {}, vision=False, secret=None,
    )


DEEPSEEK = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}, "max_tokens": 16384}


def test_default_keeps_provider_settings_and_model():
    config = provider("openai_compatible", extra_body=DEEPSEEK)
    result = with_overrides(config, None, "default")
    assert result.model == "base-model"
    assert result.extra_body == DEEPSEEK


def test_deepseek_off_disables_thinking_without_mutating_provider():
    config = provider("openai_compatible", extra_body=DEEPSEEK)
    result = with_overrides(config, "other-model", "off")
    assert result.model == "other-model"
    assert result.extra_body["chat_template_kwargs"] == {"thinking": False}
    assert result.extra_body["max_tokens"] == 16384
    assert config.extra_body["chat_template_kwargs"]["thinking"] is True


def test_deepseek_level_turns_thinking_on_with_effort():
    config = provider("openai_compatible", extra_body={"chat_template_kwargs": {"thinking": False}})
    assert with_overrides(config, None, "low").extra_body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "low"}


@pytest.mark.parametrize(
    ("kind", "auth", "level", "expected"),
    [
        ("openai", "oauth", "off", {"reasoning": {"effort": "none"}}),
        ("openai", "oauth", "medium", {"reasoning": {"effort": "medium"}}),
        ("openai", "api_key", "off", {"reasoning_effort": "none"}),
        ("openai", "api_key", "high", {"reasoning_effort": "high"}),
        ("gemini", "api_key", "low", {"reasoning_effort": "low"}),
        ("anthropic", "api_key", "medium", {"thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ],
)
def test_reasoning_level_uses_each_wire_format(kind, auth, level, expected):
    assert with_overrides(provider(kind, auth), None, level).extra_body == expected


def test_anthropic_off_removes_configured_thinking():
    config = provider("anthropic", extra_body={"thinking": {"type": "enabled", "budget_tokens": 2000}})
    assert with_overrides(config, None, "off").extra_body == {}


@pytest.mark.parametrize(
    ("config", "level"),
    [(provider("xai"), "low"), (provider("gemini"), "off"), (provider("openai_compatible"), "off")],
)
def test_unsupported_levels_are_rejected(config, level):
    assert level not in reasoning_levels(config)
    with pytest.raises(ProviderError):
        with_overrides(config, None, level)
