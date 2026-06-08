"""Tag-based release notes generator. Called by GitLab CI pipeline after a tag
is pushed on the pre-release branch.

Adapted from pr_release_notes.py (per-MR variant) — keeps the same prompt
structure (Russian, medical audience, Affine publish), but the input is a
range of commits between two tags rather than a single MR.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, date
from functools import partial
from typing import Optional

import gitlab
from jinja2 import Environment, StrictUndefined

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.token_handler import TokenHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger
from pr_agent.tools.utils.affine_publisher import publish_to_affine
from pr_agent.tools.utils.gitlab_release_publisher import create_gitlab_release, GitLabReleaseError
from pr_agent.tools.utils.telegram_publisher import TelegramPublisher, TelegramDeliveryError, escape_markdown_v2
from pr_agent.tools.utils.tg_tldr import extract_tg_tldr, replace_affine_placeholder


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


class PRReleaseNotesTag:
    def __init__(
        self,
        tag: str,
        previous_tag: str,
        project_id: int | str,
        gitlab_url: str,
        gitlab_token: str,
        ai_handler: partial[BaseAiHandler,] = LiteLLMAIHandler,
    ):
        self.tag = tag
        self.previous_tag = previous_tag
        self.project_id = project_id
        self.gitlab_url = gitlab_url
        self.gitlab_token = gitlab_token

        self.gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
        self.project = self.gl.projects.get(project_id)

        self.ai_handler = ai_handler()
        self.ai_handler.main_pr_language = "ru"

        self.primary_model = get_settings().release_notes.get("model", "claude-code/claude-opus-4-7")
        self.fallback_model = get_settings().release_notes.get("fallback_model", "claude-code/claude-opus-4-6")
        self.output_dir = get_settings().release_notes.get("output_dir", "/app/release_notes_output")
        self.affine_timeout = int(get_settings().release_notes.get("timeout_seconds_affine", 60))
        self.gitlab_release_timeout = int(get_settings().release_notes.get("gitlab_release_timeout", 30))
        self.telegram_timeout = int(get_settings().release_notes.get("telegram_timeout", 10))

        self.page_title = f"Release {self.tag} — {_format_ru_date(date.today())}"

    # ---- data collection ----

    def _collect_input_data(self) -> dict:
        prev_tag_obj = self.project.tags.get(self.previous_tag)
        prev_committed_at = prev_tag_obj.attributes["commit"]["committed_date"]

        compare = self.project.repository_compare(self.previous_tag, self.tag)
        diffs = compare.get("diffs", []) or []
        full_diff_parts = [
            f"=== FILE: {d.get('new_path') or d.get('old_path') or '<unknown>'} ===\n{d.get('diff', '') or ''}"
            for d in diffs
        ]
        full_diff = "\n\n".join(full_diff_parts)

        commits = self.project.commits.list(
            ref_name=self.tag,
            since=prev_committed_at,
            get_all=True,
        )

        commits_detailed = self._build_commit_records(commits)
        commits_summary = self._format_commits_summary(commits_detailed)
        commits_detailed_str = self._format_commits_detailed(commits_detailed)

        mrs = self.project.mergerequests.list(
            state="merged",
            target_branch="pre-release",
            updated_after=prev_committed_at,
            get_all=True,
        )
        mrs_summary = self._format_mrs_summary(mrs)

        contributors = ", ".join(sorted({c.get("author_name", "") for c in commits_detailed if c.get("author_name")}))
        release_date_str = _format_ru_date(date.today())
        try:
            prev_date_str = _format_ru_date(prev_committed_at)
        except Exception:
            prev_date_str = prev_committed_at

        compare_url = f"{self.gitlab_url.rstrip('/')}/-/compare/{self.previous_tag}...{self.tag}"

        return {
            "tag": self.tag,
            "previous_tag": self.previous_tag,
            "prev_date": prev_date_str,
            "release_date": release_date_str,
            "today_ru": release_date_str,
            "file_count": len(diffs),
            "commit_count": len(commits_detailed),
            "mr_count": len(mrs),
            "contributors": contributors or "(не определены)",
            "compare_url": compare_url,
            "commits_summary": commits_summary,
            "commits_detailed_str": commits_detailed_str,
            "mrs_summary": mrs_summary,
            "full_diff": full_diff,
            "extra_instructions": get_settings().release_notes.get("extra_instructions", {}).get("content", "") or "",
        }

    @staticmethod
    def _build_commit_records(commits) -> list[dict]:
        out = []
        for c in commits:
            attrs = getattr(c, "attributes", None) or {}
            out.append({
                "sha": attrs.get("short_id") or getattr(c, "short_id", ""),
                "full_sha": attrs.get("id") or getattr(c, "id", ""),
                "title": attrs.get("title") or getattr(c, "title", ""),
                "message": attrs.get("message") or getattr(c, "message", ""),
                "author_name": attrs.get("author_name") or getattr(c, "author_name", ""),
                "created_at": attrs.get("created_at") or getattr(c, "created_at", ""),
                "diff": "",  # populated lazily; aggregated full_diff already covers all changes
            })
        return out

    @staticmethod
    def _format_commits_summary(commits_detailed) -> str:
        lines = []
        for i, c in enumerate(commits_detailed, 1):
            lines.append(f"{i}. [{c.get('sha','?')}] {c.get('title','')} — {c.get('author_name','')}, {c.get('created_at','')}")
        return "\n".join(lines) or "(нет коммитов)"

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
            blocks.append("\n".join(block))
        return "\n\n---\n\n".join(blocks) or "(нет коммитов)"

    @staticmethod
    def _format_mrs_summary(mrs) -> str:
        if not mrs:
            return "(нет MR в этом релизе)"
        lines = []
        for mr in mrs:
            iid = getattr(mr, "iid", "?")
            title = getattr(mr, "title", "")
            author = (getattr(mr, "author", None) or {}).get("name") or "?"
            url = getattr(mr, "web_url", "")
            desc = (getattr(mr, "description", None) or "").strip()
            desc_short = desc[:300] + "…" if len(desc) > 300 else desc
            lines.append(
                f"- !{iid} «{title}» — {author}\n  URL: {url}\n  Описание: {desc_short or '(пусто)'}"
            )
        return "\n\n".join(lines)
