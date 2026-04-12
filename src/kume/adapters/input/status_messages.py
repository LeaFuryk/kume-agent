STATUS_MESSAGES = {
    "processing_media": {
        "en": "👀 Looking at your document...",
        "es": "👀 Revisando tu documento...",
    },
    "reading_analysis": {
        "en": "📄 Reading your analysis...",
        "es": "📄 Leyendo tu análisis...",
    },
    "transcribing_audio": {
        "en": "🎙️ Listening to your audio...",
        "es": "🎙️ Escuchando tu audio...",
    },
    "unsupported_media": {
        "en": "🚧 I can't process this type of file yet.",
        "es": "🚧 Todavía no puedo procesar este tipo de archivo.",
    },
    "busy": {
        "en": "⏳ I'm still working on your previous message, give me a moment...",
        "es": "⏳ Todavía estoy trabajando en tu mensaje anterior, dame un momento...",
    },
}


def get_status_message(key: str, language_code: str | None = None, **kwargs: str) -> str:
    """Get a localized status message. Falls back to English for unknown languages."""
    lang = (language_code or "en")[:2]  # "es-AR" → "es"
    messages = STATUS_MESSAGES.get(key, {})
    template = messages.get(lang, messages.get("en", key))
    return template.format(**kwargs) if kwargs else template
