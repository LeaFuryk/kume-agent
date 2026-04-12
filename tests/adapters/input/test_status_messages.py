from kume.adapters.input.status_messages import get_status_message


def test_get_status_message_english() -> None:
    result = get_status_message("processing_media", "en")
    assert result == "\U0001f440 Looking at your document..."


def test_get_status_message_spanish() -> None:
    result = get_status_message("processing_media", "es")
    assert result == "\U0001f440 Revisando tu documento..."


def test_get_status_message_unknown_language_falls_back_to_english() -> None:
    result = get_status_message("processing_media", "fr")
    assert result == "\U0001f440 Looking at your document..."


def test_get_status_message_with_kwargs_formatting() -> None:
    result = get_status_message("ingestion_complete", "en", details="3 pages extracted")
    assert result == "\u2705 Done! 3 pages extracted"


def test_get_status_message_none_language_falls_back_to_english() -> None:
    result = get_status_message("extracting_pdf", None)
    assert result == "\U0001f4c4 Extracting text..."


def test_get_status_message_language_code_with_region() -> None:
    result = get_status_message("transcribing_audio", "es-AR")
    assert result == "\U0001f399\ufe0f Transcribiendo tu audio..."


def test_get_status_message_unknown_key_returns_key() -> None:
    result = get_status_message("nonexistent_key", "en")
    assert result == "nonexistent_key"
