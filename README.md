# Kume

A multimodal AI agent that structures fragmented personal health context and food inputs into actionable nutrition guidance via Telegram.

## Architecture

Kume follows **Hexagonal Architecture (Ports & Adapters)** with [LangChain](https://github.com/langchain-ai/langchain) as the AI framework.

```
src/kume/
├── domain/          # Pure Python — entities, value objects, tool logic
├── ports/           # Abstract interfaces (LLMPort, repositories, MessagingPort)
├── services/        # OrchestratorService (agentic loop) + IngestionService (media router)
├── adapters/        # Concrete implementations
│   ├── input/       # TelegramBotAdapter (text + media handlers, status messages)
│   ├── output/      # LangChainLLMAdapter, PostgreSQL repos, Whisper STT, PDF/Audio/Image processors
│   └── tools/       # LangChain BaseTool wrappers (save, recommend, analyze)
└── infrastructure/  # Config, DI container, metrics, logging
```

The orchestrator uses an **agentic tool-use loop**: the LLM decides which tool to call, the tool executes (potentially calling other LLMs for subtasks), results flow back, and the LLM crafts the final response.

## Features

### Phase 1 — Telegram Bot + Basic Responses
- Nutrition recommendations and diet advice
- Food analysis and nutritional assessment
- Per-request metrics: token usage, cost (USD), and latency tracking
- Structured JSON logging + pretty dev formatter
- Dual-model setup: orchestrator model (e.g., gpt-4o) + cheaper tool model (e.g., gpt-4o-mini)
- Telegram HTML formatting with code block protection

### Phase 2 — Context Ingestion + Embeddings
- **Multimodal ingestion**: PDF text extraction (PyMuPDF), audio transcription (OpenAI Whisper)
- **PostgreSQL + pgvector**: Structured data (users, goals, restrictions, documents, lab markers) + vector embeddings
- **Repository pattern**: Focused repository ABCs (UserRepository, GoalRepository, etc.) with PostgreSQL adapters
- **LLMPort abstraction**: LLM as a proper port, not a framework dependency
- **Save tools**: SaveGoalTool, SaveRestrictionTool, SaveLabReportTool (LLM-powered marker extraction), SaveHealthContextTool
- **RAG context building**: ContextBuilder assembles user profile, goals, restrictions, documents, lab markers in prescribed order
- **Resource processors**: ResourceProcessorPort with PDF, Audio, and Image (stub) adapters routed by IngestionService
- **Localized status messages**: Real-time feedback during media processing (en/es)
- **Schema migrations**: Alembic with initial schema (users, goals, restrictions, documents, lab_markers)

## Getting Started

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- [Docker](https://www.docker.com/) (for PostgreSQL + pgvector)
- A Telegram bot token ([BotFather](https://t.me/botfather))
- An OpenAI API key

### Installation

```bash
git clone https://github.com/LeaFuryk/kume-agent.git
cd kume-agent
uv sync --extra dev
```

### Configuration

Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
```
TELEGRAM_TOKEN=your-telegram-bot-token
OPENAI_API_KEY=your-openai-api-key
```

Optional (with defaults):
```
ORCHESTRATOR_MODEL=gpt-4o
TOOL_MODEL=gpt-4o-mini
DATABASE_URL=postgresql+asyncpg://kume:kume@localhost:5432/kume
MAX_AGENT_ITERATIONS=5
LOG_FORMAT=pretty
LOG_LEVEL=INFO
```

### Run

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Run database migrations
uv run alembic upgrade head

# Start the bot
uv run python -m kume
```

Send the bot a message like "What should I eat for breakfast?" for nutrition advice, or send a PDF lab report to have it parsed and stored for future reference.

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

### Lint and format

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Type check

```bash
uv run mypy src/
```

### Database migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head
```

## Roadmap

- **Phase 1** — Telegram bot + basic responses (complete)
- **Phase 2** — Context ingestion + embeddings (current)
- **Phase 3** — Food image analysis
- **Phase 4** — Meal logging
- **Phase 5** — Reporting (CSV/XLSX export)

## License

This project is licensed under a **Non-Commercial License**. Free for personal, educational, and non-profit use. Commercial use requires a separate license — contact [leandrofuryk@gmail.com](mailto:leandrofuryk@gmail.com) for details.

See [LICENSE](LICENSE) for the full text.
