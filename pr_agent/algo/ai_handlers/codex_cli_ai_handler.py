import asyncio
import os
import tempfile
from pathlib import Path

import httpx
import openai
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

MODEL_RETRIES = 2
_DUMMY_REQUEST = httpx.Request("POST", "https://codex-cli/v1/chat")


class CodexCliAIHandler(BaseAiHandler):
    """AI handler that delegates completion to the local Codex CLI."""

    def __init__(self):
        self.cli_path = get_settings().get("codex_cli.cli_path", "codex")
        self.timeout = get_settings().get("codex_cli.timeout", 180)
        self.model_override = get_settings().get("codex_cli.model", "")
        self.sandbox = get_settings().get("codex_cli.sandbox", "read-only")
        self.codex_home = get_settings().get("codex_cli.codex_home", "")

    @property
    def deployment_id(self):
        return None

    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        if model.startswith("codex-cli/"):
            return model[len("codex-cli/") :]
        return model

    @retry(
        retry=retry_if_exception_type(openai.APIError) & retry_if_not_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(MODEL_RETRIES),
    )
    async def chat_completion(self, model: str, system: str, user: str,
                              temperature: float = 0.2, img_path: str = None):
        del temperature, img_path

        output_path = None
        try:
            effective_model = self.model_override or self._strip_model_prefix(model)
            prompt = f"{system}\n\n{user}" if system else user

            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as output_file:
                output_path = output_file.name

            cmd = [
                self.cli_path,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                self.sandbox,
                "--output-last-message",
                output_path,
            ]

            if effective_model:
                cmd.extend(["--model", effective_model])

            cmd.append(prompt)

            env = os.environ.copy()
            if self.codex_home:
                env["CODEX_HOME"] = self.codex_home

            get_logger().debug(f"Running Codex CLI: {' '.join(cmd[:8])}...")
            if get_settings().config.verbosity_level >= 2:
                get_logger().info(f"\nSystem prompt:\n{system}")
                get_logger().info(f"\nUser prompt:\n{user}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise openai.APIError(
                    f"Codex CLI timed out after {self.timeout}s",
                    request=_DUMMY_REQUEST,
                    body=None,
                )

            if proc.returncode != 0:
                error_text = stderr.decode("utf-8", errors="replace").strip()
                raise openai.APIError(
                    f"Codex CLI exited with code {proc.returncode}: {error_text}",
                    request=_DUMMY_REQUEST,
                    body=None,
                )

            output_text = Path(output_path).read_text(encoding="utf-8").strip() if output_path else ""
            if not output_text:
                output_text = stdout.decode("utf-8", errors="replace").strip()

            if not output_text:
                raise openai.APIError(
                    "Codex CLI returned an empty response",
                    request=_DUMMY_REQUEST,
                    body=None,
                )

            get_logger().debug(f"\nAI response:\n{output_text}")
            if get_settings().config.verbosity_level >= 2:
                get_logger().info(f"\nAI response:\n{output_text}")

            return output_text, "stop"
        except openai.APIError:
            raise
        except Exception as e:
            get_logger().warning(f"Error during Codex CLI inference: {e}")
            raise openai.APIError(
                f"Codex CLI error: {e}",
                request=_DUMMY_REQUEST,
                body=None,
            ) from e
        finally:
            if output_path:
                try:
                    Path(output_path).unlink()
                except OSError:
                    pass
