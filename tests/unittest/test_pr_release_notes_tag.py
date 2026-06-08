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
