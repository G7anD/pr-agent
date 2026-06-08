"""FastAPI router for tag-based release notes generation.

Mounted by `pr_agent.servers.gitlab_webhook` alongside the existing GitLab
webhook route. Pipeline calls this endpoint after pushing a tag to the
pre-release branch.

Auth: header `X-Auth` must equal env var `RELEASE_NOTES_SECRET`.
Idempotency: a file marker per tag in the output directory.
"""

from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, FileResponse

from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger
from pr_agent.tools.pr_release_notes_tag import PRReleaseNotesTag


router = APIRouter()


async def run_release_notes_tag(tag: str, previous_tag: str, project_id: str) -> None:
    """Background entry point — runs the tool and swallows exceptions so the
    fire-and-forget task doesn't crash the event loop."""
    try:
        tool = PRReleaseNotesTag(
            tag=tag,
            previous_tag=previous_tag,
            project_id=project_id,
            gitlab_url=get_settings().get("gitlab.url"),
            gitlab_token=get_settings().get("gitlab.personal_access_token"),
        )
        await tool.run()
    except Exception as e:
        get_logger().exception(f"release_notes_tag background task failed for {tag}: {e}")


@router.post("/generate-release-notes")
async def generate_release_notes(request: Request) -> JSONResponse:
    # 1. Auth
    auth_header = request.headers.get("X-Auth") or ""
    expected = os.environ.get(get_settings().release_notes.secret_env, "")
    if not expected or not hmac.compare_digest(auth_header, expected):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # 2. Params
    tag = request.query_params.get("tag")
    previous_tag = request.query_params.get("previous_tag")
    project_id = request.query_params.get("project_id")
    if not (tag and previous_tag and project_id):
        return JSONResponse(
            {"error": "missing required params: tag, previous_tag, project_id"},
            status_code=400,
        )

    # 3. Idempotency
    output_dir = get_settings().release_notes.output_dir
    marker = Path(output_dir) / f".published-{tag}"
    if marker.exists():
        get_logger().info(f"release_notes_tag: already published {tag}, skipping enqueue")
        return JSONResponse({"status": "already_published", "tag": tag}, status_code=200)

    # 4. Enqueue (fire-and-forget)
    asyncio.create_task(
        run_release_notes_tag(tag=tag, previous_tag=previous_tag, project_id=project_id)
    )
    return JSONResponse({"status": "accepted", "tag": tag}, status_code=202)


@router.get("/banner/{filename}")
async def serve_banner(filename: str):
    """Public (no-auth) static serve of a generated release banner PNG.

    Used by Affine and GitLab Release to render the banner image. Only serves
    `*.png` basenames from the banners dir — rejects any path-traversal.
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "bad filename"}, status_code=400)
    if not filename.endswith(".png"):
        return JSONResponse({"error": "only .png served"}, status_code=400)

    banners_dir = Path(get_settings().release_notes.output_dir) / "banners"
    path = banners_dir / filename
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(path), media_type="image/png")
