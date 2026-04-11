import html
import re

_CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to Telegram-compatible HTML.

    Handles: bold, italic, code blocks, inline code, strikethrough, and HTML escaping.
    Code block contents are preserved verbatim — inline formatting is not applied inside them.
    """
    # Extract code blocks and inline code first, replace with placeholders
    code_blocks: list[str] = []

    def _replace_code_block(match: re.Match[str]) -> str:
        code_blocks.append(f"<pre>{html.escape(match.group(2))}</pre>")
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    def _replace_inline_code(match: re.Match[str]) -> str:
        code_blocks.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    text = _CODE_BLOCK_PATTERN.sub(_replace_code_block, text)
    text = _INLINE_CODE_PATTERN.sub(_replace_inline_code, text)

    # Escape HTML in the remaining (non-code) text
    text = html.escape(text)

    # Apply inline formatting (only on non-code text)
    # Handle ***bold italic*** before ** and * to avoid crossed tags
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODEBLOCK{i}\x00", block)

    return text
