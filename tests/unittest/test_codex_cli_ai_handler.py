from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
from pr_agent.algo.ai_handlers.codex_cli_ai_handler import CodexCliAIHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


def create_codex_settings():
    return SimpleNamespace(
        config=SimpleNamespace(
            verbosity_level=0,
            get=lambda key, default=None: default,
        ),
        get=lambda key, default=None: {
            "codex_cli.cli_path": "codex",
            "codex_cli.timeout": 120,
            "codex_cli.model": "",
            "codex_cli.sandbox": "read-only",
        }.get(key, default),
    )


def create_litellm_settings():
    return SimpleNamespace(
        config=SimpleNamespace(
            reasoning_effort="medium",
            ai_timeout=120,
            custom_reasoning_model=False,
            max_model_tokens=32000,
            verbosity_level=0,
            get=lambda key, default=None: default,
        ),
        litellm=SimpleNamespace(get=lambda key, default=None: default),
        get=lambda key, default=None: default,
    )


class TestStripModelPrefix:
    def test_strips_codex_cli_prefix(self):
        assert CodexCliAIHandler._strip_model_prefix("codex-cli/gpt-5.1-codex") == "gpt-5.1-codex"

    def test_leaves_bare_model_unchanged(self):
        assert CodexCliAIHandler._strip_model_prefix("gpt-5.1-codex") == "gpt-5.1-codex"


@pytest.mark.asyncio
class TestCodexCliChatCompletion:
    @patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.get_settings")
    async def test_successful_response(self, mock_get_settings):
        mock_get_settings.return_value = create_codex_settings()
        captured = {}

        async def create_subprocess_exec(*cmd, **kwargs):
            captured["cmd"] = cmd
            output_idx = cmd.index("--output-last-message") + 1
            Path(cmd[output_idx]).write_text("AI response here", encoding="utf-8")

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            return mock_proc

        with patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.asyncio.create_subprocess_exec", new=create_subprocess_exec):
            handler = CodexCliAIHandler()
            resp, finish = await handler.chat_completion(
                model="codex-cli/gpt-5.1-codex",
                system="You are helpful",
                user="Hello",
            )

        assert resp == "AI response here"
        assert finish == "stop"
        cmd = captured["cmd"]
        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "--model" in cmd
        assert "gpt-5.1-codex" in cmd
        assert "--output-last-message" in cmd

    @patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.get_settings")
    async def test_cli_error_raises(self, mock_get_settings):
        mock_get_settings.return_value = create_codex_settings()

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"CLI error"))

        with patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            handler = CodexCliAIHandler()
            with pytest.raises(RetryError):
                await handler.chat_completion(
                    model="codex-cli/gpt-5.1-codex",
                    system="sys",
                    user="usr",
                )

    @patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.get_settings")
    async def test_timeout_raises(self, mock_get_settings):
        mock_get_settings.return_value = create_codex_settings()

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            async def raise_timeout(coro, timeout):
                coro.close()
                raise TimeoutError

            with patch("pr_agent.algo.ai_handlers.codex_cli_ai_handler.asyncio.wait_for", new=AsyncMock(side_effect=raise_timeout)):
                handler = CodexCliAIHandler()
                with pytest.raises(RetryError):
                    await handler.chat_completion(
                        model="codex-cli/gpt-5.1-codex",
                        system="sys",
                        user="usr",
                    )


@pytest.mark.asyncio
async def test_litellm_routes_codex_cli_models(monkeypatch):
    fake_settings = create_litellm_settings()
    monkeypatch.setattr(litellm_handler, "get_settings", lambda: fake_settings)

    handler = LiteLLMAIHandler()
    mock_codex_handler = MagicMock()
    mock_codex_handler.chat_completion = AsyncMock(return_value=("ok", "stop"))
    handler._codex_cli_handler = mock_codex_handler

    response = await handler.chat_completion(
        model="codex-cli/gpt-5.1-codex",
        system="system",
        user="user",
    )

    assert response == ("ok", "stop")
    mock_codex_handler.chat_completion.assert_awaited_once()
