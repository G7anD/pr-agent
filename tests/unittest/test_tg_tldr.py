import pytest

from pr_agent.tools.utils.tg_tldr import extract_tg_tldr, replace_affine_placeholder


SAMPLE_MARKDOWN = """# Aurora+ — что нового в версии 2026.06.7 (8 июня 2026)

**Общее описание:** Обновлены модули вакцинации и стоматологии.

## Основные направления
### 💉 Вакцинация
- Новое
### 🦷 Стоматология
- Улучшено

<!-- TG_TLDR_START -->
🚀 Aurora+ 2026.06.7

💉 Обновлён модуль регистрации вакцинации
🦷 Добавлена вкладка стоматологии

[Подробнее]({{ AFFINE_URL_PLACEHOLDER }})
<!-- TG_TLDR_END -->
"""


class TestExtractTgTldr:
    def test_returns_block_content_without_markers(self):
        body, tldr = extract_tg_tldr(SAMPLE_MARKDOWN)
        assert "🚀 Aurora+ 2026.06.7" in tldr
        assert "<!-- TG_TLDR_START -->" not in tldr
        assert "<!-- TG_TLDR_END -->" not in tldr
        # body must NOT contain the TLDR block (it was stripped)
        assert "<!-- TG_TLDR_START -->" not in body
        assert "🚀 Aurora+" not in body
        # body must contain the main markdown content
        assert "# Aurora+ — что нового" in body
        assert "## Основные направления" in body

    def test_missing_block_returns_none_tldr_and_unchanged_body(self):
        md = "# Some markdown\n\nNo TLDR here."
        body, tldr = extract_tg_tldr(md)
        assert tldr is None
        assert body == md

    def test_unterminated_block_returns_none_and_unchanged_body(self):
        md = "# Title\n\n<!-- TG_TLDR_START -->\nbroken without end\n"
        body, tldr = extract_tg_tldr(md)
        assert tldr is None
        assert body == md

    def test_trailing_whitespace_stripped(self):
        md = "header\n<!-- TG_TLDR_START -->\n\nhello\n\n<!-- TG_TLDR_END -->\n"
        body, tldr = extract_tg_tldr(md)
        assert tldr == "hello"


class TestReplaceAffinePlaceholder:
    def test_replaces_placeholder_with_url(self):
        text = "Foo\n[Подробнее]({{ AFFINE_URL_PLACEHOLDER }})\nBar"
        out = replace_affine_placeholder(text, "https://aff.caretech.uz/doc/abc")
        assert "{{ AFFINE_URL_PLACEHOLDER }}" not in out
        assert "https://aff.caretech.uz/doc/abc" in out

    def test_no_placeholder_returns_text_unchanged(self):
        text = "no placeholder here"
        assert replace_affine_placeholder(text, "https://x") == text

    def test_none_url_removes_placeholder_link_entirely(self):
        """When Affine failed and url is None, replace placeholder with fallback text."""
        text = "Foo\n[Подробнее]({{ AFFINE_URL_PLACEHOLDER }})\nBar"
        out = replace_affine_placeholder(text, None)
        # The placeholder line is removed entirely (no broken link in Telegram)
        assert "{{ AFFINE_URL_PLACEHOLDER }}" not in out
        assert "[Подробнее]" not in out
