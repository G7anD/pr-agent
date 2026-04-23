import copy
import os
from datetime import datetime, date
from functools import partial
from typing import Optional

from jinja2 import Environment, StrictUndefined

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.token_handler import TokenHandler
from pr_agent.config_loader import get_settings
from pr_agent.git_providers import get_git_provider
from pr_agent.git_providers.git_provider import get_main_pr_language
from pr_agent.log import get_logger
from pr_agent.tools.utils.affine_publisher import publish_to_affine


RU_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _format_ru_date(dt) -> str:
    if not dt:
        dt = datetime.utcnow()
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return dt
    return f"{dt.day} {RU_MONTHS[dt.month - 1]} {dt.year}"


class PRReleaseNotes:
    def __init__(self, pr_url: str, args: list = None,
                 ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler):
        self.git_provider = get_git_provider()(pr_url)
        self.main_language = get_main_pr_language(
            self.git_provider.get_languages(), self.git_provider.get_files()
        )
        self.ai_handler = ai_handler()
        self.ai_handler.main_pr_language = "ru"

        pr = self.git_provider.pr
        mr_number = getattr(pr, "iid", None) or getattr(pr, "number", None)
        author_info = getattr(pr, "author", None) or {}
        if isinstance(author_info, dict):
            author_name = author_info.get("name") or author_info.get("username") or "unknown"
        else:
            author_name = str(author_info)

        created_at = getattr(pr, "created_at", None)
        merged_at = getattr(pr, "merged_at", None)
        target_branch = getattr(pr, "target_branch", None) or self.git_provider.get_pr_branch()
        source_branch = getattr(pr, "source_branch", None)

        diff_files = self.git_provider.get_diff_files() or []
        full_diff = self._assemble_full_diff(diff_files)
        commit_messages_str = self.git_provider.get_commit_messages() or ""
        commit_count = self._count_commits(commit_messages_str)

        self.mr_number = mr_number
        self.pr_url_str = pr_url
        self.page_title = f"MR !{mr_number} — {_format_ru_date(date.today())}"

        self.vars = {
            "title": getattr(pr, "title", "") or "",
            "mr_number": mr_number or "",
            "author": author_name,
            "created_at": str(created_at) if created_at else "",
            "merged_at": str(merged_at) if merged_at else "(не смерджен)",
            "target_branch": target_branch or "",
            "source_branch": source_branch or "",
            "file_count": len(diff_files),
            "commit_count": commit_count,
            "commit_messages_str": commit_messages_str,
            "full_diff": full_diff,
            "extra_instructions": get_settings().pr_release_notes.get("extra_instructions", "") or "",
            "mr_url": pr_url,
            "today_ru": _format_ru_date(date.today()),
        }

        self.token_handler = TokenHandler(
            pr,
            self.vars,
            get_settings().pr_release_notes_prompt.system,
            get_settings().pr_release_notes_prompt.user,
        )

        self.primary_model = get_settings().pr_release_notes.get("model", "claude-code/claude-opus-4-7")
        self.fallback_model = get_settings().pr_release_notes.get("fallback_model", "claude-code/claude-opus-4-6")
        self.local_output_dir = get_settings().pr_release_notes.get("local_output_dir", "/app/release_notes_output")
        self.affine_timeout = int(get_settings().pr_release_notes.get("timeout_seconds_affine", 60))

    @staticmethod
    def _assemble_full_diff(diff_files) -> str:
        parts = []
        for f in diff_files:
            filename = getattr(f, "filename", None) or getattr(f, "path", None) or "<unknown>"
            patch = getattr(f, "patch", "") or ""
            parts.append(f"=== FILE: {filename} ===\n{patch}")
        return "\n\n".join(parts)

    @staticmethod
    def _count_commits(commit_messages_str: str) -> int:
        if not commit_messages_str:
            return 0
        return sum(1 for ln in commit_messages_str.splitlines() if ln.strip().startswith("-"))

    def _render_prompts(self) -> (str, str):
        env = Environment(undefined=StrictUndefined)
        system = env.from_string(get_settings().pr_release_notes_prompt.system).render(self.vars)
        user = env.from_string(get_settings().pr_release_notes_prompt.user).render(self.vars)
        return system, user

    async def _generate(self, system: str, user: str) -> Optional[str]:
        models_to_try = [self.primary_model, self.fallback_model]
        seen = set()
        last_error = None
        for model in models_to_try:
            if not model or model in seen:
                continue
            seen.add(model)
            try:
                get_logger().info(f"release_notes: generating with {model}")
                response, finish_reason = await self.ai_handler.chat_completion(
                    model=model,
                    system=system,
                    user=user,
                    temperature=0.2,
                )
                if response:
                    get_logger().info(f"release_notes: generation ok with {model} (finish_reason={finish_reason})")
                    return response
            except Exception as e:
                last_error = e
                get_logger().warning(f"release_notes: {model} failed — {e}")
                continue
        if last_error:
            raise last_error
        return None

    def _save_local(self, markdown: str) -> Optional[str]:
        try:
            os.makedirs(self.local_output_dir, exist_ok=True)
            iso = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(self.local_output_dir, f"MR-{self.mr_number}-{iso}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown)
            get_logger().info(f"release_notes: saved local copy at {path}")
            return path
        except Exception as e:
            get_logger().warning(f"release_notes: failed to save local copy — {e}")
            return None

    def _publish_success(self, affine_url: str, markdown: str, local_path: Optional[str]) -> None:
        lines = markdown.splitlines()
        intro_lines = []
        for ln in lines[:15]:
            intro_lines.append(ln)
            if len(intro_lines) >= 6 and ln.strip() == "":
                break
        intro = "\n".join(intro_lines).strip()

        local_note = f"\n\n_Локальная копия: `{local_path}`_" if local_path else ""
        comment = (
            f"✅ **Release notes опубликованы:** {affine_url}\n\n"
            f"**Краткое резюме:**\n\n{intro}"
            f"{local_note}"
        )
        self.git_provider.publish_comment(comment)

    def _publish_fallback(self, markdown: str, local_path: Optional[str], reason: str) -> None:
        local_note = f"\n\n_Локальная копия в контейнере: `{local_path}`_" if local_path else ""
        comment = (
            f"⚠️ **Загрузка в Affine не удалась** ({reason}). Полный release note ниже:\n\n"
            f"<details><summary>📝 Release notes (markdown)</summary>\n\n"
            f"{markdown}\n\n"
            f"</details>"
            f"{local_note}"
        )
        self.git_provider.publish_comment(comment)

    async def run(self):
        get_logger().info(f"release_notes: starting for {self.pr_url_str}")
        if get_settings().config.publish_output:
            self.git_provider.publish_comment(
                "📝 Abdullajon release note yozyapti... Opus 4.7 max effort bilan, bir oz vaqt oladi ⏳",
                is_temporary=True,
            )

        try:
            system, user = self._render_prompts()
            approx_tokens = self.token_handler.count_tokens(system) + self.token_handler.count_tokens(user)
            get_logger().info(f"release_notes: approx input tokens ≈ {approx_tokens}")

            markdown = await self._generate(system, user)
            if not markdown:
                self.git_provider.remove_initial_comment()
                self.git_provider.publish_comment(
                    "❌ Release note yaratib bo'lmadi — model javob bermadi. Log'larni tekshiring."
                )
                return

            local_path = self._save_local(markdown)

            try:
                self.git_provider.remove_initial_comment()
            except Exception:
                pass

            affine_url = await publish_to_affine(
                title=self.page_title,
                markdown=markdown,
                timeout=self.affine_timeout,
            )

            if affine_url:
                self._publish_success(affine_url, markdown, local_path)
            else:
                self._publish_fallback(markdown, local_path, reason="проверьте логи pr-agent")
        except Exception as e:
            get_logger().exception(f"release_notes: run failed — {e}")
            try:
                self.git_provider.remove_initial_comment()
            except Exception:
                pass
            self.git_provider.publish_comment(
                f"❌ `/release_notes` ошибка: `{e}`. Tekshiring pr-agent log'larini."
            )
