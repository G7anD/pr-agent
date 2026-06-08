from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pr_agent.tools.pr_release_notes_tag import PRReleaseNotesTag


def _fake_gitlab_client(commits=None, mrs=None, prev_tag_committed_at="2026-05-20T10:00:00Z"):
    """Build a stub python-gitlab client that returns canned data."""
    gl = MagicMock()
    proj = MagicMock()
    gl.projects.get.return_value = proj

    # tag lookup for previous tag committed date
    prev_tag = MagicMock()
    prev_tag.attributes = {"commit": {"committed_date": prev_tag_committed_at}}
    proj.tags.get.side_effect = lambda name: prev_tag

    # repository_compare returns dict with "diffs" list
    proj.repository_compare.return_value = {
        "diffs": [
            {"new_path": "file_a.py", "old_path": "file_a.py", "diff": "@@ -1 +1 @@\n-old\n+new\n"},
            {"new_path": "file_b.xml", "old_path": "file_b.xml", "diff": "@@ -1 +1 @@\n-x\n+y\n"},
        ]
    }

    # commits.list returns the commits we pass in (default = 2)
    cs = commits or [
        MagicMock(
            id="aabbccddeeff112233", short_id="aabbccdd",
            title="feat: add vaccination tab", message="feat: add vaccination tab\n\ndetails",
            author_name="Ilyos K.", created_at="2026-06-01T09:00:00Z",
        ),
        MagicMock(
            id="11223344556677889900", short_id="11223344",
            title="fix: bug in lab", message="fix: bug in lab",
            author_name="Bahrom N.", created_at="2026-06-05T15:00:00Z",
        ),
    ]
    for c in cs:
        c.attributes = {
            "id": c.id, "short_id": c.short_id, "title": c.title,
            "message": c.message, "author_name": c.author_name,
            "created_at": c.created_at,
        }
    proj.commits.list.return_value = cs

    # Stub per-commit diff fetch (used by detailed collector — returns empty diff)
    def _commit_get(sha):
        m = MagicMock()
        m.diff.return_value = []
        return m
    proj.commits.get.side_effect = _commit_get

    # MRs merged in window
    mrs_data = mrs if mrs is not None else [
        MagicMock(
            iid=123, title="Add vaccination module",
            description="Adds vaccination registration to patient card.",
            author={"name": "Ilyos K.", "username": "i.karshiboev"},
            merged_at="2026-06-01T08:55:00Z",
            web_url="https://gitlab.uicgroup.tech/.../merge_requests/123",
        ),
    ]
    proj.mergerequests.list.return_value = mrs_data

    return gl, proj


class TestPRReleaseNotesTagDataCollection:
    def test_collect_metadata_from_tags(self):
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            data = t._collect_input_data()

        assert data["tag"] == "2026.06.7"
        assert data["previous_tag"] == "2026.06.6"
        assert data["file_count"] == 2
        assert data["commit_count"] == 2
        assert data["mr_count"] == 1
        assert "Ilyos K." in data["contributors"]
        assert data["compare_url"].endswith("/compare/2026.06.6...2026.06.7")
        assert "file_a.py" in data["full_diff"]
        assert "feat: add vaccination tab" in data["commits_summary"]
        assert "Add vaccination module" in data["mrs_summary"]

    def test_handles_no_mrs_in_window(self):
        gl, proj = _fake_gitlab_client(mrs=[])
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            data = t._collect_input_data()
        assert data["mr_count"] == 0
        assert data["mrs_summary"] == "(нет MR в этом релизе)"


class TestPRReleaseNotesTagGeneration:
    @pytest.mark.asyncio
    async def test_render_prompts_substitutes_vars(self):
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            data = t._collect_input_data()
            system, user = t._render_prompts(data)

        assert "2026.06.7" in system  # tag substituted in output header instruction
        assert "что нового в версии" in system  # Russian title format present
        assert "TG_TLDR_START" in system  # TL;DR instruction present
        assert "2026.06.7" in user  # tag in metadata
        assert "Ilyos K." in user  # contributors
        assert "file_a.py" in user  # full_diff included

    @pytest.mark.asyncio
    async def test_generate_returns_markdown(self):
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )

            async def fake_completion(model, system, user, temperature):
                return ("# Aurora+ — что нового\n\n<!-- TG_TLDR_START -->\n🚀 Aurora+ 2026.06.7\n[Подробнее]({{ AFFINE_URL_PLACEHOLDER }})\n<!-- TG_TLDR_END -->", "stop")

            t.ai_handler.chat_completion = fake_completion
            md = await t._generate("sys", "usr")

        assert "Aurora+" in md
        assert "TG_TLDR_START" in md

    @pytest.mark.asyncio
    async def test_generate_falls_back_to_secondary_model(self):
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            calls = []
            async def flaky_completion(model, system, user, temperature):
                calls.append(model)
                if model == t.primary_model:
                    raise RuntimeError("primary down")
                return ("# OK", "stop")
            t.ai_handler.chat_completion = flaky_completion

            md = await t._generate("sys", "usr")

        assert md == "# OK"
        assert calls == [t.primary_model, t.fallback_model]


SAMPLE_GENERATED_MD = """# Aurora+ — что нового в версии 2026.06.7 (8 июня 2026)

**Общее описание:** Обновления в вакцинации.

## Основные направления
### 💉 Вакцинация
- Новое поле "серия"

## Рекомендации после обновления
### Для администраторов
- Дополнительных действий не требуется.

<!-- TG_TLDR_START -->
🚀 Aurora+ 2026.06.7

💉 Обновлён модуль вакцинации

[Подробнее]({{ AFFINE_URL_PLACEHOLDER }})
<!-- TG_TLDR_END -->
"""


# Import at top — needed for class-level patches below
from pr_agent.tools.utils.telegram_publisher import TelegramPublisher, TelegramDeliveryError


class TestPRReleaseNotesTagRun:
    @pytest.mark.asyncio
    async def test_run_publishes_to_all_three_destinations(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_RELEASE_CHANNEL_ID", "-1003985660672")

        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            t.output_dir = str(tmp_path)

            async def fake_completion(model, system, user, temperature):
                return (SAMPLE_GENERATED_MD, "stop")
            t.ai_handler.chat_completion = fake_completion

            calls = {"affine": 0, "gitlab": 0, "telegram": 0}

            async def fake_affine(title, markdown, timeout):
                calls["affine"] += 1
                return "https://aff.caretech.uz/doc/abc"

            def fake_gitlab_release(**kw):
                calls["gitlab"] += 1
                # Ensure description has body but not the TLDR block
                assert "TG_TLDR_START" not in kw["description"]
                assert "Aurora+ — что нового" in kw["description"]
                # Affine link injected at top
                assert "aff.caretech.uz/doc/abc" in kw["description"]
                return {"tag_name": kw["tag_name"]}

            def fake_send(self, chat_id, text, parse_mode, disable_web_page_preview):
                calls["telegram"] += 1
                # parse_mode must be MarkdownV2 per design
                assert parse_mode == "MarkdownV2", f"expected MarkdownV2, got {parse_mode}"
                # Real Affine URL replaced (not the placeholder)
                assert "AFFINE_URL_PLACEHOLDER" not in text
                # The Markdown link line is preserved unescaped, so the URL appears as-is
                assert "https://aff.caretech.uz/doc/abc" in text
                # TLDR header — "+" is a MarkdownV2 special and must appear escaped
                assert r"🚀 Aurora\+" in text, f"escaped header missing in: {text!r}"
                # Tag itself has dots — must also be escaped
                assert r"2026\.06\.7" in text
                return {"message_id": 99}

            with patch("pr_agent.tools.pr_release_notes_tag.publish_to_affine", new=fake_affine), \
                 patch("pr_agent.tools.pr_release_notes_tag.create_gitlab_release", new=fake_gitlab_release), \
                 patch.object(TelegramPublisher, "send_message", new=fake_send):
                await t.run()

        assert calls == {"affine": 1, "gitlab": 1, "telegram": 1}
        # Idempotency marker written
        assert (tmp_path / ".published-2026.06.7").exists()
        # Local markdown backup written
        assert (tmp_path / "2026.06.7.md").exists()

    @pytest.mark.asyncio
    async def test_run_continues_when_telegram_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
        monkeypatch.setenv("TELEGRAM_RELEASE_CHANNEL_ID", "-1003985660672")
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            t.output_dir = str(tmp_path)

            async def fake_completion(model, system, user, temperature):
                return (SAMPLE_GENERATED_MD, "stop")
            t.ai_handler.chat_completion = fake_completion

            async def fake_affine(title, markdown, timeout):
                return "https://aff.caretech.uz/doc/abc"
            def fake_gitlab_release(**kw):
                return {"tag_name": kw["tag_name"]}
            def fake_send(self, chat_id, text, parse_mode, disable_web_page_preview):
                raise TelegramDeliveryError("chat not found")

            with patch("pr_agent.tools.pr_release_notes_tag.publish_to_affine", new=fake_affine), \
                 patch("pr_agent.tools.pr_release_notes_tag.create_gitlab_release", new=fake_gitlab_release), \
                 patch.object(TelegramPublisher, "send_message", new=fake_send):
                # Should NOT raise
                await t.run()

        # Marker still written — Telegram is best-effort
        assert (tmp_path / ".published-2026.06.7").exists()

    @pytest.mark.asyncio
    async def test_run_skips_when_marker_exists(self, tmp_path):
        gl, proj = _fake_gitlab_client()
        with patch("pr_agent.tools.pr_release_notes_tag.gitlab.Gitlab", return_value=gl):
            t = PRReleaseNotesTag(
                tag="2026.06.7", previous_tag="2026.06.6", project_id=42,
                gitlab_url="https://gitlab.uicgroup.tech",
                gitlab_token="tok",
            )
            t.output_dir = str(tmp_path)
            (tmp_path / ".published-2026.06.7").write_text("")

            ai_called = []
            async def fake_completion(*a, **kw):
                ai_called.append(1)
                return ("nope", "stop")
            t.ai_handler.chat_completion = fake_completion

            await t.run()
        assert ai_called == [], "AI must not be called when marker exists"


class TestInsertBannerAfterH1:
    def test_inserts_image_after_h1(self):
        from pr_agent.tools.pr_release_notes_tag import _insert_banner_after_h1
        md = "# Aurora+ — версия 26.6.1\n\n**Общее описание:** текст\n"
        out = _insert_banner_after_h1(md, "https://pra.caretech.uz/banner/26.6.1.png")
        lines = out.split("\n")
        assert lines[0] == "# Aurora+ — версия 26.6.1"
        assert "![Aurora+ release](https://pra.caretech.uz/banner/26.6.1.png)" in out
        assert out.index("![Aurora+ release]") < out.index("**Общее описание:**")

    def test_prepends_when_no_h1(self):
        from pr_agent.tools.pr_release_notes_tag import _insert_banner_after_h1
        md = "no header here\njust text"
        out = _insert_banner_after_h1(md, "https://x/b.png")
        assert out.startswith("![Aurora+ release](https://x/b.png)")
        assert "no header here" in out

    def test_only_first_h1_gets_banner(self):
        from pr_agent.tools.pr_release_notes_tag import _insert_banner_after_h1
        md = "# First\ntext\n# Second\nmore"
        out = _insert_banner_after_h1(md, "https://x/b.png")
        assert out.count("![Aurora+ release]") == 1
        assert out.index("![Aurora+ release]") < out.index("# Second")
