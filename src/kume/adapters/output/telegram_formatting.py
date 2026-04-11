import html
import re


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to Telegram-compatible HTML.

    Handles: bold, italic, code blocks, inline code, and HTML escaping.
    """
    # Escape HTML special characters first
    text = html.escape(text)

    # Code blocks (``` ... ```) → <pre>
    text = re.sub(r"```(\w*)\n(.*?)```", r"<pre>\2</pre>", text, flags=re.DOTALL)

    # Inline code (` ... `) → <code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Bold (**text**) → <b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Italic (*text*) → <i>
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)

    # Strikethrough (~~text~~) → <s>
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    return text
