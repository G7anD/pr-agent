"""Minimal GitLab Releases API client — stdlib-only.

Why stdlib: pr-agent already pulls python-gitlab transitively but its
`releases` API surface is awkward to mock and the JSON shape is stable.
A 30-line stdlib helper is easier to test and reason about.

Endpoint: POST /api/v4/projects/:id/releases
Docs: https://docs.gitlab.com/api/releases/
"""

from __future__ import annotations

import json
from typing import Optional
from urllib import request, error


class GitLabReleaseError(RuntimeError):
    pass


def create_gitlab_release(
    gitlab_url: str,
    access_token: str,
    project_id: int | str,
    tag_name: str,
    name: str,
    description: str,
    timeout: int = 30,
    released_at: Optional[str] = None,
) -> dict:
    """Create a GitLab Release object. Returns the parsed JSON response.

    Raises GitLabReleaseError on any HTTP or network failure (including 409
    if the release already exists — caller may want to ignore that).
    """
    url = f"{gitlab_url.rstrip('/')}/api/v4/projects/{project_id}/releases"
    payload = {
        "tag_name": tag_name,
        "name": name,
        "description": description,
    }
    if released_at:
        payload["released_at"] = released_at
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={
            "PRIVATE-TOKEN": access_token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except error.HTTPError as e:
        raise GitLabReleaseError(f"HTTP {e.code}: {e.reason}") from e
    except Exception as e:
        raise GitLabReleaseError(str(e)) from e
