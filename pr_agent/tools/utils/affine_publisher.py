"""Thin wrapper around the `affine` CLI binary (github.com/tomohiro-owada/affine-cli)
to publish markdown docs to a self-hosted Affine workspace.

Returns the public/shareable URL (or doc URL constructed from id) on success,
or None on any failure - caller is expected to fall back to inline output.

Expected environment variables:
    AFFINE_BASE_URL        - e.g. "https://aff.caretech.uz"
    AFFINE_API_TOKEN       - user access token (from `affine token generate --name X`)
    AFFINE_WORKSPACE_ID    - target workspace id
    AFFINE_CLI             - (optional) override path to the `affine` binary (default: "affine")

Safety note: uses asyncio.create_subprocess_exec with an argv list (argv-form,
equivalent to posix execve). No shell is spawned, no user input is
interpolated into a command string, so this is NOT vulnerable to shell
injection.
"""

import asyncio
import asyncio.subprocess as _aiosub
import json
import os
import re
import shutil
from typing import Optional, Tuple

from pr_agent.log import get_logger


_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


def _find_url(text: str, base_url: Optional[str] = None) -> Optional[str]:
    if not text:
        return None
    candidates = _URL_RE.findall(text)
    if not candidates:
        return None
    if base_url:
        host = base_url.rstrip("/")
        for u in candidates:
            if host in u:
                return u
    return candidates[-1]


def _extract_doc_id(text: str) -> Optional[str]:
    if not text:
        return None
    # JSON output path
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("id", "docId", "doc_id"):
                if key in data and isinstance(data[key], str):
                    return data[key]
    except Exception:
        pass
    # UUID-like fallback
    m = re.search(r'"(?:id|docId|doc_id)"\s*:\s*"([a-zA-Z0-9_-]{8,})"', text)
    if m:
        return m.group(1)
    return None


def _resolve_env() -> Tuple[str, str, str, str, str]:
    """Returns (base, token, workspace, cli, public_base).

    `base` is the URL affine-cli uses to talk to Affine (often an internal/docker
    network URL like http://172.17.0.1:3010). `public_base` is what we embed in
    the MR comment for humans to click (e.g. https://aff.caretech.uz). If
    AFFINE_PUBLIC_URL is unset, it defaults to AFFINE_BASE_URL.
    """
    base = os.environ.get("AFFINE_BASE_URL", "").strip()
    token = (
        os.environ.get("AFFINE_API_TOKEN", "").strip()
        or os.environ.get("AFFINE_TOKEN", "").strip()
    )
    workspace = os.environ.get("AFFINE_WORKSPACE_ID", "").strip()
    cli = os.environ.get("AFFINE_CLI", "affine-cli").strip()
    public_base = os.environ.get("AFFINE_PUBLIC_URL", "").strip() or base
    return base, token, workspace, cli, public_base


async def publish_to_affine(title: str, markdown: str, timeout: int = 60) -> Optional[str]:
    logger = get_logger()
    base, token, workspace, cli, public_base = _resolve_env()

    if not (base and token and workspace):
        logger.info(
            "affine_publisher: AFFINE_BASE_URL/AFFINE_API_TOKEN/AFFINE_WORKSPACE_ID "
            "not all set - skipping Affine upload"
        )
        return None

    if shutil.which(cli) is None and not os.path.isabs(cli):
        logger.warning(f"affine_publisher: `{cli}` not found in PATH")
        return None

    env = {
        **os.environ,
        "AFFINE_API_TOKEN": token,
        "AFFINE_BASE_URL": base,
        "AFFINE_WORKSPACE_ID": workspace,
    }
    argv = [
        cli, "doc", "create-from-markdown",
        "--title", title,
        "--content", markdown,
        "--workspace", workspace,
    ]

    short_argv = [argv[0]] + argv[1:5] + ["...", "--workspace", workspace[:8] + "..."]
    logger.info(f"affine_publisher: spawning: {' '.join(short_argv)}")

    try:
        # argv-list form (no shell, no string interpolation) - safe API
        proc = await _aiosub.create_subprocess_exec(
            *argv,
            stdin=_aiosub.DEVNULL,
            stdout=_aiosub.PIPE,
            stderr=_aiosub.PIPE,
            env=env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:
                pass
            logger.warning(f"affine_publisher: timed out after {timeout}s")
            return None

        stdout = (stdout_b or b"").decode("utf-8", errors="replace").strip()
        stderr = (stderr_b or b"").decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            logger.warning(
                f"affine_publisher: affine exit={proc.returncode} stderr={stderr[:500]!r}"
            )
            return None

        # Always prefer constructing the URL from doc_id against the *public*
        # base, because stdout URLs (if any) will use the internal API URL.
        url = None
        doc_id = _extract_doc_id(stdout)
        if doc_id:
            url = f"{public_base.rstrip('/')}/workspace/{workspace}/{doc_id}"
            logger.info(f"affine_publisher: constructed public URL: {url}")
        if not url:
            url = _find_url(stdout, base_url=public_base) or _find_url(stderr, base_url=public_base)
        if not url:
            logger.warning(
                f"affine_publisher: no URL / doc_id in affine output. "
                f"stdout={stdout[:300]!r} stderr={stderr[:300]!r}"
            )
            return None

        logger.info(f"affine_publisher: success url={url}")
        return url
    except FileNotFoundError as e:
        logger.warning(f"affine_publisher: binary not found - {e}")
        return None
    except Exception as e:
        logger.warning(f"affine_publisher: unexpected error - {e}")
        return None
