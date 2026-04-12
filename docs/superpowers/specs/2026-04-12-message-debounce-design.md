# Message Debounce & Batching — Design Spec

## Overview

Replace per-message processing with a debounce/batching system. All messages from the same user within a 3-second silence window are collected into a batch, processed together, and produce one combined response.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Debounce window | 3 seconds of silence | Enough for multi-message typing and PDF attachment, short enough to feel responsive |
| Batching scope | One queue per user, all message types | Text + PDFs + audio all combine into one orchestrator call |
| PDF extraction | Parallel within the batch | PDFs are independent and heavy — parallel extraction then combine |
| Audio/text | Concatenated in order | Preserves conversation flow |
| Response | ONE per batch | No more crossed responses or duplicate greetings |
| Anticipatory messages | System prompt fallback | If "Here are my results" arrives alone (PDFs took >3s), LLM responds with "Send them over!" |

## Architecture

### MessageBatcher

New class in `src/kume/adapters/input/message_batcher.py`:

```python
class PendingBatch:
    """Accumulates messages for one user within the debounce window."""
    texts: list[str]           # text messages in order
    media: list[MediaItem]     # (raw_bytes, mime_type, caption)
    timer: asyncio.TimerHandle | None

@dataclass
class MediaItem:
    raw_bytes: bytes
    mime_type: str
    caption: str

class MessageBatcher:
    """Collects messages per user and fires batch processing after silence."""
    
    def __init__(
        self,
        debounce_seconds: float = 3.0,
        on_batch_ready: Callable[[int, PendingBatch], Awaitable[None]],
    ) -> None: ...

    async def add_text(self, telegram_id: int, text: str) -> None:
        """Add a text message. Resets the debounce timer."""

    async def add_media(self, telegram_id: int, item: MediaItem) -> None:
        """Add a media item. Resets the debounce timer."""
```

When the timer fires (3s silence), it calls `on_batch_ready(telegram_id, batch)`.

### Updated TelegramBotAdapter

Simplified — both handlers just add to the batcher:

```python
class TelegramBotAdapter:
    def __init__(self, ..., batcher: MessageBatcher) -> None: ...

    async def handle_message(self, update, context) -> None:
        # Validate, then:
        await self._batcher.add_text(telegram_id, text)

    async def handle_media(self, update, context) -> None:
        # Download file, then:
        await self._batcher.add_media(telegram_id, MediaItem(raw_bytes, mime_type, caption))
```

No more per-user locks — the batcher handles sequencing.

### Batch Processing (on_batch_ready callback)

```python
async def process_batch(telegram_id: int, batch: PendingBatch) -> None:
    # 1. Send status message
    await messaging.send_message(chat_id, "📄 Reading your messages...")

    # 2. Extract all PDFs in parallel
    pdf_tasks = [ingestion.process(m.raw_bytes, m.mime_type) 
                 for m in batch.media if m.mime_type == "application/pdf"]
    audio_tasks = []
    for m in batch.media:
        if m.mime_type.startswith("audio/"):
            audio_tasks.append(ingestion.process(m.raw_bytes, m.mime_type))

    pdf_results = await asyncio.gather(*pdf_tasks) if pdf_tasks else []
    audio_results = []
    for task in audio_tasks:
        audio_results.append(await task)  # sequential for audio

    # 3. Combine everything in order
    parts = []
    for text in batch.texts:
        parts.append(text)
    for caption, extracted in zip(pdf_captions, pdf_results):
        if caption:
            parts.append(caption)
        parts.append(extracted)
    for result in audio_results:
        parts.append(result)

    combined = "\n\n".join(parts)

    # 4. Truncate if needed
    if len(combined) > MAX_EXTRACTED_TEXT:
        combined = combined[:MAX_EXTRACTED_TEXT] + "\n\n[Truncated]"

    # 5. ONE orchestrator call → ONE response
    response = await orchestrator.process(telegram_id, combined)
    await messaging.send_message(chat_id, response)
```

### System Prompt Addition

Add to the existing system prompt:

```
## Anticipatory Messages

If the user sends a message that clearly precedes files they haven't sent yet
(like "Here are my results", "Check these out", "Sending you my labs"), and there's 
no actual data attached, respond briefly acknowledging you're ready:
"Send them over! I'm ready to take a look 👀"

Don't try to analyze or respond substantively to setup messages with no content.
```

### Status Messages

Update for batch context:

```python
STATUS_MESSAGES = {
    "processing_batch": {
        "en": "📄 Reading your messages...",
        "es": "📄 Leyendo tus mensajes...",
    },
    "processing_single_pdf": {
        "en": "📄 Reading your analysis...",
        "es": "📄 Leyendo tu análisis...",
    },
}
```

Use `processing_batch` when batch has >1 item, `processing_single_pdf` for single PDF.

## Data Flow

```
User sends: "Estos son mis análisis" → batcher.add_text() → timer starts (3s)
User sends: PDF1                      → batcher.add_media() → timer resets (3s)
User sends: PDF2                      → batcher.add_media() → timer resets (3s)
User sends: PDF3                      → batcher.add_media() → timer resets (3s)
3 seconds silence                     → on_batch_ready fires:
                                         → "📄 Leyendo tus mensajes..."
                                         → Extract PDF1, PDF2, PDF3 in parallel
                                         → Combine: "Estos son mis análisis\n\n[PDF1]\n\n[PDF2]\n\n[PDF3]"
                                         → ONE orchestrator call
                                         → ONE comparative response
```

## Edge Cases

- **Single text message**: debounce fires after 3s, processes normally (like today but with 3s delay)
- **Single PDF**: same as single text — 3s delay then process
- **Text arrives >3s after PDFs**: separate batch, LLM responds to text using saved context from earlier PDFs
- **Anticipatory message + slow attachment**: if "Here are my results" alone triggers (>3s), LLM says "Send them over!" — not a bug, good UX

## Files Changed

- `src/kume/adapters/input/message_batcher.py` — NEW: MessageBatcher, PendingBatch, MediaItem
- `src/kume/adapters/input/telegram_bot.py` — SIMPLIFIED: handlers add to batcher, batch callback does processing
- `src/kume/adapters/input/status_messages.py` — batch-aware messages
- `src/kume/services/orchestrator.py` — system prompt: anticipatory message guidance
- `src/kume/infrastructure/container.py` — wire batcher

## Testing

- Test debounce timer resets on new message
- Test batch fires after 3s silence
- Test PDFs extract in parallel within batch
- Test single message still works (with 3s delay)
- Test combined text + PDF batch produces one orchestrator call
- Test anticipatory message alone gets "send them over" response (system prompt test)
