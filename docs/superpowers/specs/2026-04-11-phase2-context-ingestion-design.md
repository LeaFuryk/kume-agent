# Phase 2: Context Ingestion + Embeddings — Design Spec

## Overview

Phase 2 adds persistence (PostgreSQL + pgvector), multimodal document ingestion (PDF, audio, text), and RAG retrieval to Kume. Users can send health documents, voice notes, and text to build their personal health context. The existing tools (`ask_recommendation`, `analyze_food`) are upgraded to retrieve and use this context when responding.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL + pgvector | Single DB for structured data + vector embeddings. PRD recommendation. Relational needs (user → goals → documents → lab markers) |
| Dev setup | Docker Compose + env var | `docker compose up` for local dev, connection string for production |
| ORM | SQLAlchemy async + Alembic | Mature, typed, async support via asyncpg. Alembic for migrations. LangChain PGVector uses it |
| PDF parsing | PyMuPDF text extraction + LLM structured extraction | Lab reports are digital PDFs — text extraction is reliable. LLM extracts typed LabMarker entities from the text |
| Audio | LangChain Whisper integration behind SpeechToTextPort | Consistent with LangChain ecosystem. Handles Spanish well |
| Chunking | LangChain SemanticChunker | Splits by meaning not fixed size. Better retrieval quality |
| RAG approach | Hybrid — structured data (PostgreSQL) + semantic search (pgvector) | Structured for precision ("What was my cholesterol?"), semantic for recall ("Am I eating healthy?") |
| Embeddings | OpenAI text-embedding-3-small via LangChain | LangChain's `Embeddings` base class is the port, `OpenAIEmbeddings` is the adapter |
| Vector store | LangChain PGVector | Same PostgreSQL instance, `VectorStore` base class is the port |
| Image handling | Skip in Phase 2 | PRD puts food images in Phase 3. Food photos are the primary use case for LogMealTool |
| Media routing | Telegram adapter layer, not orchestrator | Adapter handles PDF/audio/text routing. Orchestrator only sees processed text |
| Logging | Pretty formatter for dev, JSON for production | `LOG_FORMAT=pretty` for human-readable reasoning chain, `LOG_FORMAT=json` for structured production logs |

## Architecture

### Media Routing

```
TelegramBotAdapter
  ├── text only → OrchestratorService(text)
  └── any media (PDF, audio, image) → IngestionService.process(bytes, mime_type)
                                     → returns extracted text
                                     → OrchestratorService(user_caption + extracted text)
```

The `TelegramBotAdapter` doesn't know about resource types — it just detects "not text" and delegates to `IngestionService`.

### IngestionService (media router)

Routes raw bytes to the correct `ResourceProcessorPort` adapter by mime type:

```python
class IngestionService:
    def __init__(self, processors: dict[str, ResourceProcessorPort]) -> None: ...

    async def process(self, raw_bytes: bytes, mime_type: str) -> str:
        processor = self.processors.get(mime_type)
        if not processor:
            raise UnsupportedMediaType(mime_type)
        return await processor.process(raw_bytes)
```

No business logic, no database access, no LLM calls. Just mime-type routing to the right processor.

### ResourceProcessorPort + Adapters

```python
# ports/output/resource_processor.py
class ResourceProcessorPort(ABC):
    async def process(self, raw_bytes: bytes) -> str: ...

# adapters/output/pdf_processor.py
class PDFProcessor(ResourceProcessorPort):
    """Extracts text from PDF using PyMuPDF."""

# adapters/output/audio_processor.py
class AudioProcessor(ResourceProcessorPort):
    """Transcribes audio using SpeechToTextPort (Whisper)."""
    def __init__(self, stt: SpeechToTextPort) -> None: ...

# adapters/output/image_processor.py (stub — Phase 3)
class ImageProcessor(ResourceProcessorPort):
    """Stub — returns 'not supported yet'."""
```

Container wires processors by mime type:
```python
def ingestion_service(self) -> IngestionService:
    return IngestionService(processors={
        "application/pdf": PDFProcessor(),
        "audio/ogg": AudioProcessor(stt=self.stt_port()),
        "audio/mpeg": AudioProcessor(stt=self.stt_port()),
        "image/jpeg": ImageProcessor(),   # stub
        "image/png": ImageProcessor(),    # stub
    })
```

### Orchestrator Tools (all decisions happen here)

The orchestrator's LLM sees the full text (user message + extracted content from PDF/audio) and picks the right tool:

```
OrchestratorService (agentic loop — single decision-maker)
  → LLM receives text (may include extracted PDF/audio content)
  → LLM picks tool:
      ├── save_goal → saves Goal to DatabasePort
      ├── save_restriction → saves Restriction to DatabasePort
      ├── save_lab_report → extracts LabMarkers via LLM, saves to DatabasePort,
      │                     chunks + embeds in pgvector
      ├── save_health_context → saves Document to DatabasePort,
      │                         chunks + embeds in pgvector
      ├── ask_recommendation → builds RAG context, calls LLM
      ├── analyze_food → builds RAG context, calls LLM
      └── (casual chat — no tool, responds directly)
```

This keeps the orchestrator as the single decision-maker. Classification, structured extraction, database writes, and embedding all happen through tool calls within the agentic loop.

### RAG Context Building

```
ContextBuilder.build(user_id, query)
  │
  ├── 1. User profile          ← DatabasePort.get_or_create_user()
  ├── 2. Goals                  ← DatabasePort.get_goals(user_id)
  ├── 3. Restrictions           ← DatabasePort.get_restrictions(user_id)
  ├── 4. Relevant documents     ← VectorStore.similarity_search(query, filter={user_id})
  ├── 5. Recent lab markers     ← DatabasePort.get_lab_markers(user_id, last_3_months)
  └── 6. Current input          ← the user's message
  
  → Assembled into a single context string in this exact order (CLAUDE.md rule #7)
```

Tools (`ask_recommendation`, `analyze_food`) call the context builder before calling the LLM. The orchestrator doesn't change.

## Domain Layer

### New Entities

```python
@dataclass(frozen=True)
class Goal:
    id: str
    user_id: str
    description: str
    created_at: datetime

@dataclass(frozen=True)
class Restriction:
    id: str
    user_id: str
    type: str            # "allergy", "intolerance", "diet"
    description: str
    created_at: datetime

@dataclass(frozen=True)
class Document:
    id: str
    user_id: str
    type: str            # "lab_report", "diet_plan", "medical_report"
    filename: str
    summary: str
    ingested_at: datetime

@dataclass(frozen=True)
class LabMarker:
    id: str
    document_id: str
    user_id: str
    name: str            # "COLESTEROL TOTAL"
    value: float
    unit: str            # "mg/dL"
    reference_range: str # "< 200 mg/dL"
    date: datetime
```

### Context Builder

`domain/context.py` — Assembles RAG context using injected callables (no port imports):

```python
def build_context(
    user_id: str,
    query: str,
    get_goals: Callable,
    get_restrictions: Callable,
    get_lab_markers: Callable,
    search_documents: Callable,
) -> str:
    """Assemble RAG context in the prescribed order."""
```

### Updated/New Tool Handlers

- `ask_recommendation` and `analyze_food` — Updated signatures to accept a `context: str` parameter assembled by the context builder
- `save_goal` (NEW) — Saves a goal via injected database callable
- `save_restriction` (NEW) — Saves a restriction via injected database callable
- `save_lab_report` (NEW) — Extracts LabMarkers from text via LLM callable, saves via database callable, chunks + embeds via vector store callable
- `save_health_context` (NEW) — Saves document via database callable, chunks + embeds via vector store callable
- `ingest_context` — REMOVED (replaced by the 4 specific tools above)

## Ports Layer

### DatabasePort (custom ABC)

```python
class DatabasePort(ABC):
    async def get_or_create_user(self, telegram_id: int) -> User: ...
    async def save_document(self, doc: Document) -> None: ...
    async def save_lab_markers(self, markers: list[LabMarker]) -> None: ...
    async def get_lab_markers(self, user_id: str, name: str | None = None, since: datetime | None = None) -> list[LabMarker]: ...
    async def save_goal(self, goal: Goal) -> None: ...
    async def get_goals(self, user_id: str) -> list[Goal]: ...
    async def save_restriction(self, restriction: Restriction) -> None: ...
    async def get_restrictions(self, user_id: str) -> list[Restriction]: ...
```

### SpeechToTextPort (custom ABC)

```python
class SpeechToTextPort(ABC):
    async def transcribe(self, audio_bytes: bytes, language: str = "es") -> str: ...
```

### LangChain Ports (no custom ABCs)

- `Embeddings` base class → `OpenAIEmbeddings` adapter
- `VectorStore` base class → `PGVector` adapter

## Services Layer

### IngestionService (NEW — media router)

```python
class IngestionService:
    def __init__(self, processors: dict[str, ResourceProcessorPort]) -> None: ...

    async def process(self, raw_bytes: bytes, mime_type: str) -> str:
        """Route raw bytes to the correct processor by mime type. Returns extracted text."""
```

No database access, no LLM calls, no classification. Routes to the right `ResourceProcessorPort` adapter.

### OrchestratorService (UNCHANGED)

Only sees text. Tools are updated to handle database/vector operations, but the orchestrator itself doesn't change.

## Adapters Layer

### PostgresAdapter (DatabasePort implementation)

- SQLAlchemy async ORM models in `adapters/output/postgres_models.py`
- Adapter in `adapters/output/postgres_db.py` implementing `DatabasePort`
- All queries scoped to `user_id` (CLAUDE.md rule #6)

### WhisperAdapter (SpeechToTextPort implementation)

- Uses LangChain's OpenAI Whisper integration
- In `adapters/output/whisper_stt.py`

### Resource Processor Adapters

- `adapters/output/pdf_processor.py` — `PDFProcessor(ResourceProcessorPort)` using PyMuPDF
- `adapters/output/audio_processor.py` — `AudioProcessor(ResourceProcessorPort)` using SpeechToTextPort
- `adapters/output/image_processor.py` — `ImageProcessor(ResourceProcessorPort)` stub for Phase 3

### Updated TelegramBotAdapter

Adds a media handler for non-text messages:
- `MessageHandler(filters.Document | filters.VOICE | filters.AUDIO | filters.PHOTO, handle_media)` — downloads file, gets mime type, calls `IngestionService.process(bytes, mime_type)`, forwards extracted text + caption to OrchestratorService
- Text messages still route to OrchestratorService directly (unchanged)

### Updated/New Tool Adapters

- `AskRecommendationTool` and `AnalyzeFoodTool` — inject context builder, call it before domain handler to include RAG context
- `SaveGoalTool` (NEW) — saves a Goal to DatabasePort
- `SaveRestrictionTool` (NEW) — saves a Restriction to DatabasePort
- `SaveLabReportTool` (NEW) — LLM extracts LabMarkers from text, saves Document + LabMarkers to DatabasePort, chunks + embeds in pgvector
- `SaveHealthContextTool` (NEW) — saves Document to DatabasePort, chunks + embeds in pgvector
- `IngestContextTool` — REMOVED (replaced by the 4 specific tools above)

## Infrastructure

### Docker Compose

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: kume
      POSTGRES_USER: kume
      POSTGRES_PASSWORD: kume
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

### Settings (new env vars)

```
DATABASE_URL=postgresql+asyncpg://kume:kume@localhost:5432/kume
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
LOG_FORMAT=pretty  # or "json" for production
```

### Alembic Migrations

```
alembic/
├── alembic.ini
├── env.py
└── versions/
    └── 001_initial_schema.py   # users, goals, restrictions, documents, lab_markers
```

### Container (new factory methods)

```python
def database_port(self) -> DatabasePort
def vector_store(self) -> PGVector
def embeddings(self) -> OpenAIEmbeddings
def stt_port(self) -> SpeechToTextPort
def ingestion_service(self) -> IngestionService    # processors dict keyed by mime type
def context_builder(self) -> ContextBuilder         # needs database_port + vector_store
```

### Metrics (extended)

New metric types:

```python
@dataclass(frozen=True)
class EmbeddingMetric:
    model: str
    token_count: int
    chunk_count: int
    cost_usd: float
    latency_ms: float

@dataclass(frozen=True)
class IngestionMetric:
    document_type: str       # "pdf", "audio", "text"
    chunks_created: int
    lab_markers_extracted: int
    total_latency_ms: float
```

`RequestMetrics` extended with `embeddings: list[EmbeddingMetric]` and `ingestions: list[IngestionMetric]`.

### Pretty Logging

New `PrettyFormatter` for development:

```
── Request 7a3f ── telegram_id=12345 ──────────────────
│ 📦 Context: 3 goals, 2 restrictions, 5 chunks retrieved
│ 🤖 LLM call: gpt-4o | 342 in → 156 out | $0.0041 | 1.2s
│   └─ Tool: ask_recommendation
│      🤖 LLM call: gpt-4o-mini | 891 in → 203 out | $0.0003 | 0.8s
│ ✅ Total: $0.0044 | 2.1s | 1233 tokens in | 359 tokens out
───────────────────────────────────────────────────────
```

Controlled by `LOG_FORMAT` env var. Default: `pretty` for dev, `json` for production.

## Dependencies (new)

```toml
dependencies = [
    # existing...
    "asyncpg>=0.30",
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.14",
    "langchain-postgres>=0.0.12",
    "pymupdf>=1.24",
]
```

## Testing Strategy

- **Domain tests:** ContextBuilder assembly order, ingest_context extraction logic (mock LLM)
- **Port tests:** DatabasePort and SpeechToTextPort ABC enforcement
- **Adapter tests:**
  - PostgresAdapter: CRUD for all entities, user_id scoping (real PostgreSQL via Docker)
  - WhisperAdapter: mocked LangChain Whisper
  - Updated tool adapters: context builder integration
- **Service tests:** IngestionService — mock ports, verify PDF/audio/text paths
- **Integration tests:** Full ingestion flow (mock LLM + real PostgreSQL)
- **Telegram adapter tests:** PDF handler, voice handler, routing logic
- **Test database:** Same Docker Compose, separate `kume_test` database, transactions rolled back per test

## User Feedback Messages

Status messages sent to the user during media processing, localized based on `update.effective_user.language_code` from Telegram:

```python
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
```

Language detected from `update.effective_user.language_code`. Falls back to `"en"` if unknown. Simple dict lookup — no LLM calls for status messages.

The `TelegramBotAdapter` sends these via `MessagingPort` at each processing step, giving the user real-time feedback during media ingestion.

## Out of Scope (Phase 3+)

- Food image analysis (Phase 3 — LogMealTool with VisionPort)
- Meal logging persistence (Phase 4)
- Report generation (Phase 5)
- Conversation history / multi-turn memory
- Reranking in retrieval
