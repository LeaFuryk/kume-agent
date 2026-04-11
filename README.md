# Kume

A multimodal AI agent that structures fragmented personal health context and food inputs into actionable nutrition guidance via Telegram.

## Architecture

Kume follows **Hexagonal Architecture (Ports & Adapters)** with [LangChain](https://github.com/langchain-ai/langchain) as the AI framework.

```
src/kume/
├── domain/          # Pure Python — entities, value objects, tool logic
├── ports/           # Abstract interfaces (MessagingPort)
├── services/        # OrchestratorService — agentic tool-use loop
├── adapters/        # Concrete implementations
│   ├── input/       # TelegramBotAdapter
│   ├── output/      # TelegramMessagingAdapter
│   └── tools/       # LangChain BaseTool wrappers
└── infrastructure/  # Config, DI container, metrics, logging
```

The orchestrator uses an **agentic tool-use loop**: the LLM decides which tool to call, the tool executes (potentially calling other LLMs for subtasks), results flow back, and the LLM crafts the final response.

## Features

- Nutrition recommendations and diet advice
- Food analysis and nutritional assessment
- Per-request metrics: token usage, cost (USD), and latency tracking
- Structured JSON logging with request correlation
- Dual-model setup: orchestrator model (e.g., gpt-4o) + cheaper tool model (e.g., gpt-4o-mini)

## Getting Started

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
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

Edit `.env` with your actual values:

```
TELEGRAM_TOKEN=your-telegram-bot-token
OPENAI_API_KEY=your-openai-api-key
```

### Run

```bash
uv run python -m kume
```

The bot starts polling for Telegram messages. Send it a message like "What should I eat for breakfast?" and it will respond with personalized nutrition advice.

## Development

### Run tests

```bash
uv run pytest tests/ -v
```

### Lint

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Type check

```bash
uv run mypy src/
```

## Roadmap

- **Phase 1** — Telegram bot + basic responses (current)
- **Phase 2** — Context ingestion + embeddings (RAG)
- **Phase 3** — Food image analysis
- **Phase 4** — Meal logging
- **Phase 5** — Reporting (CSV/XLSX export)

## License

This project is licensed under a **Non-Commercial License**. Free for personal, educational, and non-profit use. Commercial use requires a separate license — contact [leandrofuryk@gmail.com](mailto:leandrofuryk@gmail.com) for details.

See [LICENSE](LICENSE) for the full text.
