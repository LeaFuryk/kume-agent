# Kume

A multimodal AI agent that structures fragmented personal health context and food inputs into actionable nutrition guidance via Telegram.

## Architecture

Kume follows **Hexagonal Architecture (Ports & Adapters)** with [LangChain](https://github.com/langchain-ai/langchain) as the AI framework.

```
src/kume/
├── domain/          # Pure Python — entities, value objects, tool logic, session resolution
├── ports/           # Abstract interfaces (LLMPort, VisionPort, repositories, MessagingPort)
├── services/        # OrchestratorService (agentic loop + session history) + IngestionService
├── adapters/        # Concrete implementations
│   ├── input/       # TelegramBotAdapter (text + media handlers, message batching)
│   ├── output/      # LangChainLLM, OpenAI Vision, PostgreSQL repos, Whisper STT, processors
│   └── tools/       # LangChain BaseTool wrappers (save, recommend, analyze, log meal)
└── infrastructure/  # Config, DI container, metrics, logging, session/image stores
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

### Phase 3 — Food Image Analysis + Meal Logging
- **Food image analysis**: Send a food photo and get a detailed nutritional breakdown (calories, protein, carbs, fat, fiber, sodium, sugar, saturated fat, cholesterol) via OpenAI Vision
- **Meal logging**: Log meals with full nutritional details — from images or text descriptions. Optional timestamp for retroactive logging
- **Intent-based flow**: LLM decides whether to analyze-only ("is this healthy?") or analyze+log ("I just ate this") based on user intent
- **Portion confirmation**: Kume estimates portion size and asks for confirmation before logging
- **Conversation history**: In-memory session history with 1-hour silence gap detection — enables multi-turn flows like analyze → correct portion → log
- **VisionPort abstraction**: Vision AI as a proper port with OpenAI adapter (same pattern as LLMPort)
- **Meals in RAG context**: Recent meals included in ContextBuilder so recommendations account for what the user already ate
- **Schema migration**: Alembic 002 adds `meals` table with all nutritional columns

## Usage Examples

Here's what you can do with Kume via Telegram:

### Get nutrition advice
```
You: What should I eat for breakfast if I want to lower my triglycerides?
Kume: Great question! Here are some options that help lower triglycerides:
      • Oatmeal with berries and walnuts (~350 kcal)
      • Greek yogurt with flaxseed (~200 kcal)
      • Avocado toast on whole grain (~300 kcal)
      All rich in omega-3s and fiber 💪
```

### Send a lab report (PDF)
```
You: [sends PDF] Here are my latest lab results
Kume: 📄 Reading your analysis...
      I extracted 12 markers from your report. Here's what stands out:
      • Triglycerides: 180 mg/dL (ref: <150) — slightly elevated ⚠️
      • LDL: 125 mg/dL (ref: <130) — borderline, keep watching
      • HDL: 55 mg/dL — good range ✅
      Overall you're trending in the right direction!
```

### Analyze a food photo
```
You: [sends photo] Is this healthy?
Kume: Looks like a Caesar salad with grilled chicken (~450 kcal).
      • Protein: 35g ✅
      • Carbs: 18g
      • Fat: 28g (saturated: 6g from dressing/cheese)
      Good protein source! The dressing adds saturated fat — ask for it
      on the side next time. Aligns well with your goal to eat healthier 👍
```

### Log a meal from a photo
```
You: [sends photo] Just had this for lunch
Kume: Looks like ~2 slices of pepperoni pizza (~550 kcal). Does that sound right?
You: It was 3 slices actually
Kume: Got it! Logged: 3 slices pepperoni pizza
      • 825 kcal | 33g protein | 87g carbs | 39g fat
      • Sodium: 1800mg — that's high for one meal
      Try balancing dinner with something lighter and low-sodium 🥗
```

### Log a meal from text
```
You: I had a grilled chicken salad for dinner at 7pm
Kume: Meal logged: grilled chicken salad
      • 380 kcal | 35g protein | 12g carbs | 20g fat
      Great choice for dinner — high protein, low carb 💪
```

### Set health goals
```
You: I want to lower my triglycerides and lose 5kg
Kume: Goals saved! I'll personalize all my advice around:
      • Lower triglycerides
      • Lose 5kg
      Send me your lab results and I'll track your progress 📊
```

### Send a voice note
```
You: [voice note] "I'm allergic to shellfish and I try to avoid gluten"
Kume: 🎙️ Listening to your audio...
      Got it! Saved:
      • Allergy: shellfish
      • Diet: avoids gluten
      I'll factor these into every recommendation from now on.
```

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
VISION_MODEL=gpt-4o
DATABASE_URL=postgresql+asyncpg://kume:kume@localhost:5432/kume
MAX_AGENT_ITERATIONS=5
LOG_FORMAT=pretty
LOG_LEVEL=INFO
```

Optional (LangSmith observability):
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=kume
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

### LLM Evaluations

Kume includes an eval framework for verifying LLM behavior: tool selection, intent classification (log vs analyze), and response quality. Eval cases are defined in YAML files under `tests/evals/cases/`.

```bash
# Validate eval cases load correctly (no API key needed)
uv run pytest tests/evals/ -v

# Run evals against real LLM (requires OPENAI_API_KEY)
uv run pytest tests/evals/ -m eval -v

# Run with a different model (A/B testing)
EVAL_MODEL=gpt-5-mini uv run pytest tests/evals/ -m eval -v

# Run a specific eval suite
uv run pytest tests/evals/test_tool_selection.py -m eval -v
```

**31 eval cases across 3 suites:**
- **Tool selection** (15 cases) — verifies correct tool dispatch for all 12 tools
- **Intent classification** (10 cases) — verifies log vs analyze-only decisions
- **Response quality** (6 cases) — LLM-as-judge scoring on language, tone, conciseness

Without `OPENAI_API_KEY`, only YAML structure validation runs. With the key, cases execute against a real LLM orchestrator and assert tool selection, intent decisions, and response quality scores (>= 3/5).

### LangSmith Observability

Kume supports [LangSmith](https://smith.langchain.com) for tracing, debugging, and monitoring LLM calls in production.

**Setup:** Add `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT=kume` to your `.env`. That's it — LangChain auto-traces every agent call.

**What you get:**
- Visual trace timeline for every request (system prompt → tool calls → response)
- Token usage, latency, and cost per LLM call
- Tool call arguments and results
- Error traces for debugging failures
- Free tier: 5,000 traces/month

**Dashboard:** Visit [smith.langchain.com](https://smith.langchain.com) to view traces after sending messages to the bot.

## Roadmap

- **Phase 1** — Telegram bot + basic responses ✅
- **Phase 2** — Context ingestion + embeddings ✅
- **Phase 3** — Food image analysis + meal logging ✅
- **Phase 5** — Reporting (CSV/XLSX export)

## License

This project is licensed under a **Non-Commercial License**. Free for personal, educational, and non-profit use. Commercial use requires a separate license — contact [leandrofuryk@gmail.com](mailto:leandrofuryk@gmail.com) for details.

See [LICENSE](LICENSE) for the full text.
