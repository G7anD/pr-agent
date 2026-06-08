"""Telegram TL;DR extraction from release-notes markdown.

The release-notes prompt asks Claude to append a block of the form:

    <!-- TG_TLDR_START -->
    ...short Telegram message...
    [Подробнее]({{ AFFINE_URL_PLACEHOLDER }})
    <!-- TG_TLDR_END -->

This module extracts that block (stripping the markers and surrounding
whitespace) and returns the body without the block. It also replaces
the `{{ AFFINE_URL_PLACEHOLDER }}` token with the real Affine page URL
once it is known (or removes the placeholder link entirely if Affine
publishing failed).
"""

import re
from typing import Optional, Tuple


_TLDR_BLOCK_RE = re.compile(
    r"\n?<!--\s*TG_TLDR_START\s*-->(?P<body>.*?)<!--\s*TG_TLDR_END\s*-->\n?",
    re.DOTALL,
)

# Matches a whole "[text]({{ AFFINE_URL_PLACEHOLDER }})" line including its trailing newline
_PLACEHOLDER_LINE_RE = re.compile(
    r"\n?\[[^\]]*\]\(\s*\{\{\s*AFFINE_URL_PLACEHOLDER\s*\}\}\s*\)\n?"
)

_PLACEHOLDER_RE = re.compile(r"\{\{\s*AFFINE_URL_PLACEHOLDER\s*\}\}")


def extract_tg_tldr(markdown: str) -> Tuple[str, Optional[str]]:
    """Return (body_without_tldr, tldr_content) or (markdown, None) if no block found."""
    m = _TLDR_BLOCK_RE.search(markdown)
    if not m:
        return markdown, None
    body = (markdown[: m.start()] + markdown[m.end():]).rstrip() + "\n"
    tldr = m.group("body").strip()
    return body, tldr


def replace_affine_placeholder(text: str, affine_url: Optional[str]) -> str:
    """Replace `{{ AFFINE_URL_PLACEHOLDER }}` with the real URL.

    If `affine_url` is None (Affine publishing failed):
    - Remove the entire `[text]({{ AFFINE_URL_PLACEHOLDER }})` link line so no broken
      link ships to Telegram.
    - Also scrub any remaining bare placeholders that appear outside a link wrapper,
      so the literal Jinja-style token never reaches downstream destinations.
    """
    if affine_url is None:
        without_link = _PLACEHOLDER_LINE_RE.sub("", text)
        without_bare = _PLACEHOLDER_RE.sub("", without_link)
        if without_bare == text:
            return text
        return without_bare.rstrip() + "\n"
    return _PLACEHOLDER_RE.sub(affine_url, text)
