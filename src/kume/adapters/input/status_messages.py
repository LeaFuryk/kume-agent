STATUS_MESSAGES = {
    "processing_media": {
        "en": "👀 Looking at your document...",
        "es": "👀 Revisando tu documento...",
    },
    "extracting_pdf": {
        "en": "📄 Extracting text...",
        "es": "📄 Extrayendo texto...",
    },
    "transcribing_audio": {
        "en": "🎙️ Transcribing your audio...",
        "es": "🎙️ Transcribiendo tu audio...",
    },
    "ingestion_complete": {
        "en": "✅ Done! {details}",
        "es": "✅ ¡Listo! {details}",
    },
    "unsupported_media": {
        "en": "🚧 I can't process this type of file yet.",
        "es": "🚧 Todavía no puedo procesar este tipo de archivo.",
    },
}


def get_status_message(key: str, language_code: str | None = None, **kwargs: str) -> str:
    """Get a localized status message. Falls back to English for unknown languages."""
    lang = (language_code or "en")[:2]  # "es-AR" → "es"
    messages = STATUS_MESSAGES.get(key, {})
    template = messages.get(lang, messages.get("en", key))
    return template.format(**kwargs) if kwargs else template
