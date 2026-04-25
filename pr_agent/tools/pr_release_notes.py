import copy
import os
import re
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

MAX_PER_COMMIT_DIFF_CHARS = 200_000   # hard safety cap per commit; expect most to be small
MAX_COMMITS_WITH_DIFF = 30            # most recent N commits get per-commit diff; older get summary only
DIFF_FETCH_PARALLELISM = 8            # concurrent GitLab API calls when fetching per-commit diffs


def _format_ru_date(dt) -> str:
    if not dt:
        dt = datetime.utcnow()
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return dt
    return f"{dt.day} {RU_MONTHS[dt.month - 1]} {dt.year}"


# pr-agent's _handle_request auto-injects this when config.response_language is set
# (e.g. "uz-UZ" for Uzbek). For /release_notes the output MUST be Russian, so we strip
# the auto-injection before rendering the prompt.
_LANG_INJECTION_RE = re.compile(
    r"(?:\n*={3,}\n*\n?In addition,\s*)?Your response MUST be written in the language "
    r"corresponding to locale code:\s*'[^']*'\.\s*This is crucial\.\s*",
    re.DOTALL,
)


def _strip_lang_injection(extra_instructions: str) -> str:
    if not extra_instructions:
        return ""
    cleaned = _LANG_INJECTION_RE.sub("", extra_instructions).strip()
    return cleaned


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
        mr_description = (getattr(pr, "description", None) or "").strip()

        # Bypass pr-agent truncation: go direct to GitLab changes() API
        full_diff, file_count = self._collect_full_mr_diff_untruncated()
        commits_detailed = self._collect_commits_detailed()
        commits_summary = self._commits_summary(commits_detailed)
        commits_detailed_str = self._format_commits_detailed(commits_detailed)

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
            "file_count": file_count,
            "commit_count": len(commits_detailed),
            "mr_description": mr_description or "(описание не заполнено)",
            "commits_summary": commits_summary,
            "commits_detailed_str": commits_detailed_str,
            "full_diff": full_diff,
            "extra_instructions": _strip_lang_injection(
                get_settings().pr_release_notes.get("extra_instructions", "") or ""
            ),
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

    def _collect_full_mr_diff_untruncated(self):
        """Fetch full MR diff without pr-agent's large-patch truncation. Returns (diff_str, file_count)."""
        try:
            raw_changes = self.git_provider.mr.changes().get("changes", []) or []
            parts = []
            for c in raw_changes:
                fname = c.get("new_path") or c.get("old_path") or "<unknown>"
                diff = c.get("diff", "") or ""
                parts.append(f"=== FILE: {fname} ===\n{diff}")
            return "\n\n".join(parts), len(raw_changes)
        except Exception as e:
            get_logger().warning(f"release_notes: mr.changes() failed, falling back to get_diff_files — {e}")
            try:
                diff_files = self.git_provider.get_diff_files() or []
                parts = []
                for f in diff_files:
                    filename = getattr(f, "filename", None) or getattr(f, "path", None) or "<unknown>"
                    patch = getattr(f, "patch", "") or ""
                    parts.append(f"=== FILE: {filename} ===\n{patch}")
                return "\n\n".join(parts), len(diff_files)
            except Exception as e2:
                get_logger().warning(f"release_notes: fallback also failed — {e2}")
                return "", 0

    def _collect_commits_detailed(self):
        """List of dicts: sha, title, message, author_name, created_at, diff (full per-commit).

        Caps per-commit diff fetching to MAX_COMMITS_WITH_DIFF most recent commits to keep
        latency bounded (each fetch is a serial GitLab API call). Uses a ThreadPoolExecutor
        for concurrent fetches. Older commits get title+body but no diff (aggregated MR diff
        still includes all their line-level changes).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out = []
        try:
            commits_iter = list(self.git_provider.mr.commits())
        except Exception as e:
            get_logger().warning(f"release_notes: mr.commits() failed — {e}")
            return out

        project = None
        try:
            # Prefer numeric project_id from the MR object; the string namespace
            # path (self.id_project) sometimes 404s due to URL-encoding quirks.
            project_ident = getattr(self.git_provider.mr, "project_id", None) or self.git_provider.id_project
            project = self.git_provider.gl.projects.get(project_ident)
        except Exception as e:
            get_logger().warning(f"release_notes: project lookup failed — {e}")

        # First pass: extract metadata for all commits (no API calls)
        base_records = []
        for c in commits_iter:
            try:
                attrs = getattr(c, "attributes", None) or {}
                sha = attrs.get("id") or getattr(c, "id", "")
                short_id = attrs.get("short_id") or getattr(c, "short_id", sha[:8] if sha else "")
                base_records.append({
                    "sha": short_id,
                    "full_sha": sha,
                    "title": attrs.get("title") or getattr(c, "title", ""),
                    "message": attrs.get("message") or getattr(c, "message", ""),
                    "author_name": attrs.get("author_name") or getattr(c, "author_name", ""),
                    "created_at": attrs.get("created_at") or getattr(c, "created_at", ""),
                    "diff": "",
                })
            except Exception as e:
                get_logger().warning(f"release_notes: commit parsing failed — {e}")
                continue

        # Decide which commits to fetch diffs for: most-recent first, capped.
        # mr.commits() returns newest-first by default in GitLab.
        to_fetch = [r for r in base_records[:MAX_COMMITS_WITH_DIFF] if r["full_sha"]]
        older_count = max(0, len(base_records) - len(to_fetch))
        if older_count:
            get_logger().info(
                f"release_notes: fetching per-commit diffs for {len(to_fetch)} most recent "
                f"commits; {older_count} older commits will be listed without diff "
                f"(full aggregated MR diff still covers them)"
            )

        if project and to_fetch:
            def _fetch(rec):
                try:
                    commit_obj = project.commits.get(rec["full_sha"])
                    diffs = commit_obj.diff(get_all=True) or []
                    parts = []
                    running = 0
                    for d in diffs:
                        fname = d.get("new_path") or d.get("old_path") or "<unknown>"
                        patch = d.get("diff", "") or ""
                        entry = f"=== FILE: {fname} ===\n{patch}"
                        if running + len(entry) > MAX_PER_COMMIT_DIFF_CHARS:
                            parts.append("(... остальные файлы коммита пропущены из-за размера ...)")
                            break
                        parts.append(entry)
                        running += len(entry)
                    rec["diff"] = "\n\n".join(parts)
                except Exception as e:
                    get_logger().warning(f"release_notes: per-commit diff fetch failed for {rec['sha']} — {e}")
                return rec

            with ThreadPoolExecutor(max_workers=DIFF_FETCH_PARALLELISM) as ex:
                list(ex.map(_fetch, to_fetch))

        # Return in ORIGINAL (newest-first) order — matches how user/author thinks
        return base_records

    @staticmethod
    def _commits_summary(commits_detailed) -> str:
        lines = []
        for i, c in enumerate(commits_detailed, 1):
            lines.append(f"{i}. [{c.get('sha','?')}] {c.get('title','')} — {c.get('author_name','')}, {c.get('created_at','')}")
        return "\n".join(lines)

    @staticmethod
    def _format_commits_detailed(commits_detailed) -> str:
        blocks = []
        for i, c in enumerate(commits_detailed, 1):
            block = [
                f"### Коммит {i}: [{c.get('sha','?')}] {c.get('title','')}",
                f"**Автор:** {c.get('author_name','')}  •  **Дата:** {c.get('created_at','')}",
                "",
                "**Полное тело коммита:**",
                c.get("message", "") or "(пусто)",
            ]
            diff = c.get("diff", "")
            if diff:
                block.extend(["", "**Diff коммита:**", "```diff", diff, "```"])
            blocks.append("\n".join(block))
        return "\n\n---\n\n".join(blocks)

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
            get_logger().warning(f"release_notes: failed to save local copy - {e}")
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
            f"📝 **Release notes готовы** ({reason}). Содержимое ниже:\n\n"
            f"<details><summary>Развернуть release note (markdown)</summary>\n\n"
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
            get_logger().info(
                f"release_notes: input sizes - files={self.vars['file_count']} "
                f"commits={self.vars['commit_count']} tokens≈{approx_tokens}"
            )

            markdown = await self._generate(system, user)
            if not markdown:
                try: self.git_provider.remove_initial_comment()
                except Exception: pass
                self.git_provider.publish_comment(
                    "❌ Release note yaratib bo'lmadi - model javob bermadi. Log'larni tekshiring."
                )
                return

            local_path = self._save_local(markdown)

            try: self.git_provider.remove_initial_comment()
            except Exception: pass

            affine_url = await publish_to_affine(
                title=self.page_title,
                markdown=markdown,
                timeout=self.affine_timeout,
            )

            if affine_url:
                self._publish_success(affine_url, markdown, local_path)
            else:
                reason = "Affine не настроен" if not os.environ.get("AFFINE_API_TOKEN") else "ошибка при загрузке в Affine, см. логи"
                self._publish_fallback(markdown, local_path, reason=reason)
        except Exception as e:
            get_logger().exception(f"release_notes: run failed - {e}")
            try: self.git_provider.remove_initial_comment()
            except Exception: pass
            self.git_provider.publish_comment(
                f"❌ `/release_notes` ошибка: `{e}`. Tekshiring pr-agent log'larini."
            )
