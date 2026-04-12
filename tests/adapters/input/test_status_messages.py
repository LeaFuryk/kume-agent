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


def test_get_status_message_busy_english() -> None:
    result = get_status_message("busy", "en")
    assert result == "\u23f3 I'm still working on your previous message, give me a moment..."


def test_get_status_message_none_language_falls_back_to_english() -> None:
    result = get_status_message("reading_analysis", None)
    assert result == "\U0001f4c4 Reading your analysis..."


def test_get_status_message_language_code_with_region() -> None:
    result = get_status_message("transcribing_audio", "es-AR")
    assert result == "\U0001f399\ufe0f Escuchando tu audio..."


def test_get_status_message_unknown_key_returns_key() -> None:
    result = get_status_message("nonexistent_key", "en")
    assert result == "nonexistent_key"
