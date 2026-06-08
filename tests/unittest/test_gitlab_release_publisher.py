import json
from unittest.mock import patch, MagicMock

import pytest

from pr_agent.tools.utils.gitlab_release_publisher import (
    create_gitlab_release,
    GitLabReleaseError,
)


class TestCreateGitLabRelease:
    def test_posts_to_releases_endpoint(self):
        with patch("pr_agent.tools.utils.gitlab_release_publisher.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 201
            mock_resp.read.return_value = json.dumps(
                {"name": "Aurora+ 2026.06.7", "tag_name": "2026.06.7",
                 "_links": {"self": "https://gitlab.uicgroup.tech/api/v4/projects/42/releases/2026.06.7"}}
            ).encode()
            urlopen.return_value.__enter__.return_value = mock_resp

            result = create_gitlab_release(
                gitlab_url="https://gitlab.uicgroup.tech",
                access_token="tok123",
                project_id=42,
                tag_name="2026.06.7",
                name="Aurora+ — что нового в версии 2026.06.7",
                description="# Markdown body",
            )

            assert result["tag_name"] == "2026.06.7"
            # Verify URL + headers + body
            req = urlopen.call_args[0][0]
            assert req.full_url == "https://gitlab.uicgroup.tech/api/v4/projects/42/releases"
            assert req.headers["Private-token"] == "tok123"
            assert req.headers["Content-type"] == "application/json"
            payload = json.loads(req.data.decode("utf-8"))
            assert payload["tag_name"] == "2026.06.7"
            assert payload["name"].startswith("Aurora+")
            assert payload["description"] == "# Markdown body"

    def test_raises_on_http_error(self):
        from urllib.error import HTTPError
        with patch("pr_agent.tools.utils.gitlab_release_publisher.request.urlopen") as urlopen:
            urlopen.side_effect = HTTPError(
                url="x", code=409, msg="Conflict",
                hdrs=None, fp=None,
            )
            with pytest.raises(GitLabReleaseError, match="409"):
                create_gitlab_release(
                    gitlab_url="https://gitlab.uicgroup.tech",
                    access_token="tok123",
                    project_id=42,
                    tag_name="2026.06.7",
                    name="x",
                    description="x",
                )
