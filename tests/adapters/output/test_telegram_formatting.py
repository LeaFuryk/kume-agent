from kume.adapters.output.telegram_formatting import markdown_to_telegram_html


def test_bold_conversion() -> None:
    assert markdown_to_telegram_html("**hello**") == "<b>hello</b>"


def test_italic_conversion() -> None:
    assert markdown_to_telegram_html("*hello*") == "<i>hello</i>"


def test_inline_code_conversion() -> None:
    assert markdown_to_telegram_html("`code`") == "<code>code</code>"


def test_strikethrough_conversion() -> None:
    assert markdown_to_telegram_html("~~deleted~~") == "<s>deleted</s>"


def test_html_escaping() -> None:
    result = markdown_to_telegram_html("Use <html> & 'quotes'")
    assert "&lt;html&gt;" in result
    assert "&amp;" in result


def test_bold_not_confused_with_italic() -> None:
    result = markdown_to_telegram_html("**bold** and *italic*")
    assert "<b>bold</b>" in result
    assert "<i>italic</i>" in result


def test_plain_text_unchanged() -> None:
    assert markdown_to_telegram_html("hello world") == "hello world"


def test_code_block_conversion() -> None:
    text = "```python\nprint('hi')\n```"
    result = markdown_to_telegram_html(text)
    assert "<pre>" in result
    assert "print" in result


def test_code_block_contents_not_reformatted() -> None:
    """Markdown inside code blocks should NOT be converted to HTML formatting."""
    text = "```\n**not bold** and *not italic*\n```"
    result = markdown_to_telegram_html(text)
    assert "<b>" not in result
    assert "<i>" not in result
    assert "**not bold**" in result or "&ast;&ast;" in result or "**" in result


def test_inline_code_contents_not_reformatted() -> None:
    """Markdown inside inline code should NOT be converted."""
    result = markdown_to_telegram_html("`**not bold**`")
    assert "<b>" not in result
    assert "<code>" in result


def test_code_block_html_escaped() -> None:
    """HTML in code blocks should be escaped."""
    text = "```\n<div>test</div>\n```"
    result = markdown_to_telegram_html(text)
    assert "&lt;div&gt;" in result
