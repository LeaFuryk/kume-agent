# Phase 1: Telegram Bot + Basic Responses — Design Spec

## Overview

Phase 1 delivers a working Telegram bot that receives text messages, classifies user intent via LLM tool calling, executes the appropriate tool (which may itself call LLMs for subtasks), and returns a natural language response. The architecture follows hexagonal (ports & adapters) with LangChain as the AI framework.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Intent classification | LLM tool calling (not string parsing) | LLM decides which tool to invoke directly — aligns with PRD's "controlled tool execution" |
| Tool execution model | Agentic tool-use loop (Option C) | Standard pattern: LLM calls tool → execute → result back → LLM crafts final response |
| Tool internals | Tools can call LLMs for subtasks | Same adapter, possibly different/cheaper model. Enables agentic sub-resolution |
| Orchestrator placement | Domain owns tools, service owns loop (Approach C) | Keeps domain pure, orchestration is an application concern |
| AI framework | LangChain | Provides LLM abstractions, tool calling, agent loops, callbacks for metrics |
| Metrics strategy | In-memory collector + structured JSON logging | Per-request cost/latency breakdown without external infra |
| Phase 1 functional tools | `ask_recommendation`, `analyze_food` | Work with LLM only, no database/vector store needed |
| Phase 1 stub tools | `ingest_context`, `log_meal`, `request_report` | Return "coming soon" — need Phase 2+ backends |

## Architecture

### Layer Mapping

```
src/kume/
├── domain/              # Pure Python. Entities, value objects, tool logic.
│   ├── entities.py      # User
│   ├── value_objects.py # ConfidenceScore, TokenUsage-related types
│   ├── tools/           # Tool handler logic (pure functions/classes)
│   └── metrics.py       # RequestMetrics, LLMCallMetric, ToolExecutionMetric
├── ports/
│   └── output/
│       └── messaging.py # MessagingPort ABC (our own — LangChain doesn't cover this)
├── services/
│   └── orchestrator.py  # OrchestratorService — owns the agentic tool-use loop
├── adapters/
│   ├── input/
│   │   └── telegram_bot.py      # TelegramBotAdapter — receives messages
│   ├── output/
│   │   └── telegram_messaging.py # TelegramMessagingAdapter — sends replies
│   └── tools/                    # BaseTool subclasses wrapping domain tool logic
│       ├── ask_recommendation.py
│       ├── analyze_food.py
│       ├── ingest_context.py     # stub
│       ├── log_meal.py           # stub
│       └── request_report.py     # stub
└── infrastructure/
    ├── config.py         # Settings dataclass, env loading
    ├── container.py      # DI wiring
    ├── logging.py        # Structured JSON logging setup
    └── metrics.py        # MetricsCollector + MetricsCallbackHandler
```

### LangChain ↔ Hexagonal Mapping

| Hexagonal Layer | LangChain Concept | Concrete Example |
|---|---|---|
| Ports (output) | `BaseChatModel` | LLM interface — already abstract |
| Ports (output) | `MessagingPort` | Our own ABC (LangChain doesn't cover Telegram) |
| Adapters (output) | `ChatOpenAI` | Concrete LLM adapter from `langchain-openai` |
| Domain | Tool logic as pure Python | No LangChain imports in domain |
| Adapters (bridge) | `BaseTool` subclasses | Wrap domain tool handlers for LangChain |
| Services | `AgentExecutor` | Orchestrator uses LangChain agent primitives |
| Infrastructure | `BaseCallbackHandler` | `MetricsCallbackHandler` captures token/cost/latency |

### Data Flow

```
Telegram message (text)
  → TelegramBotAdapter.handle_message(update, context)
    → OrchestratorService.process(telegram_id, text)
      → MetricsCollector.start_request()
      → Build messages: [SystemMessage(persona), HumanMessage(text)]
      → AgentExecutor.invoke() with tools + MetricsCallbackHandler
        ↕ LangChain tool-use loop:
          → LLM (orchestrator_model) selects tool + arguments
          → BaseTool adapter calls domain tool logic
            → Tool may call LLM (tool_model) internally for subtask
          → ToolMessage result fed back to orchestrator LLM
          → LLM generates final response (or calls another tool)
      → MetricsCollector.end_request()
      → Log RequestMetrics as structured JSON
      → Return response text
    → MessagingPort.send_message(chat_id, response)
```

## Domain Layer

### Entities

```python
@dataclass(frozen=True)
class User:
    id: str
    telegram_id: int
```

Minimal for Phase 1. Extended with goals, restrictions, etc. in Phase 2.

### Metrics Types

```python
@dataclass(frozen=True)
class LLMCallMetric:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    purpose: str              # "orchestrator", "tool:ask_recommendation", etc.

@dataclass(frozen=True)
class ToolExecutionMetric:
    tool_name: str
    latency_ms: float
    success: bool

@dataclass
class RequestMetrics:
    request_id: str
    telegram_id: int
    start_time: datetime
    end_time: datetime | None
    llm_calls: list[LLMCallMetric]
    tool_executions: list[ToolExecutionMetric]

    @property
    def total_cost_usd(self) -> float: ...

    @property
    def total_latency_ms(self) -> float: ...

    @property
    def total_input_tokens(self) -> int: ...

    @property
    def total_output_tokens(self) -> int: ...
```

### Domain Tool Logic

Tool handlers in `domain/tools/` are pure Python. They define what each tool does without LangChain dependency:

- `ask_recommendation.py` — Takes a query string and optional context. Returns nutrition recommendation text. May accept an LLM-calling function for sub-resolution (injected, not imported).
- `analyze_food.py` — Takes a food description. Returns analysis text. Same LLM sub-resolution pattern.
- `ingest_context.py`, `log_meal.py`, `request_report.py` — Return stub messages.

The domain tool logic receives an LLM-calling callable (a plain `Callable`) via DI, not a `BaseChatModel` directly. This keeps LangChain out of the domain.

## Ports Layer

### MessagingPort (our own ABC)

```python
class MessagingPort(ABC):
    @abstractmethod
    async def send_message(self, chat_id: int, text: str) -> None: ...
```

### LLM Port

No custom ABC. LangChain's `BaseChatModel` serves as the LLM port. Adapters implement it (e.g., `ChatOpenAI`). The orchestrator and tool adapters receive `BaseChatModel` via DI.

## Services Layer

### OrchestratorService

```python
class OrchestratorService:
    def __init__(
        self,
        llm: BaseChatModel,              # orchestrator model
        tools: list[BaseTool],
        metrics_collector: MetricsCollector,
        system_prompt: str,
        max_iterations: int = 5,
    ): ...

    async def process(self, telegram_id: int, text: str) -> str:
        """Full agentic loop: classify → tool call → respond."""
        ...
```

- Agent is constructed once in `__init__` via `create_tool_calling_agent` + `AgentExecutor`
- Each `process()` call creates a fresh `MetricsCallbackHandler` (request-scoped) and passes it via the `callbacks` argument to `AgentExecutor.invoke()`
- System prompt establishes Kume as a nutrition-focused AI assistant, describes available tools
- Returns the final text response
- If loop exceeds `max_iterations`, returns a fallback "I wasn't able to process that" message
- Tool errors are returned as `ToolMessage` so the LLM informs the user gracefully

### System Prompt

The system prompt for the orchestrator:
- Establishes Kume's identity as a personal nutrition AI agent
- Instructs the LLM to use tools when the user asks about food, nutrition, or health
- Instructs the LLM to respond conversationally for greetings, small talk, or off-topic messages (no tool call needed)
- Lists tool capabilities at a high level so the LLM knows when each applies

## Adapters Layer

### Driving Adapter — TelegramBotAdapter

```python
class TelegramBotAdapter:
    def __init__(
        self,
        orchestrator: OrchestratorService,
        messaging: MessagingPort,
    ): ...

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        telegram_id = update.effective_user.id
        text = update.message.text
        response = await self.orchestrator.process(telegram_id, text)
        await self.messaging.send_message(update.effective_chat.id, response)
```

Phase 1: text messages only. Non-text messages get a "I can only handle text messages for now" reply.

### Driven Adapter — TelegramMessagingAdapter

```python
class TelegramMessagingAdapter(MessagingPort):
    def __init__(self, bot: Bot): ...

    async def send_message(self, chat_id: int, text: str) -> None:
        await self.bot.send_message(chat_id=chat_id, text=text)
```

### Tool Adapters

Each `BaseTool` subclass in `adapters/tools/` wraps a domain tool handler:

```python
class AskRecommendationTool(BaseTool):
    name: str = "ask_recommendation"
    description: str = "Get personalized nutrition recommendations based on the user's question about diet, meals, or health goals"
    llm: BaseChatModel                    # tool_model, injected via DI

    def _run(self, query: str) -> str:
        # Calls domain logic, passing LLM as a callable for sub-resolution
        return ask_recommendation(query, llm_call=self._call_llm)

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
```

Stub tools return a fixed message without calling any LLM.

## Infrastructure Layer

### Settings

```python
@dataclass(frozen=True)
class Settings:
    telegram_token: str
    openai_api_key: str
    orchestrator_model: str          # default: "gpt-4o"
    tool_model: str                  # default: "gpt-4o-mini"
    max_agent_iterations: int        # default: 5
    log_level: str                   # default: "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load from environment variables. Raises ValueError for missing required vars."""
        ...
```

### Container (DI)

No framework — explicit factory methods:

```python
class Container:
    def __init__(self, settings: Settings): ...

    def orchestrator_llm(self) -> BaseChatModel:
        return ChatOpenAI(model=self.settings.orchestrator_model, api_key=...)

    def tool_llm(self) -> BaseChatModel:
        return ChatOpenAI(model=self.settings.tool_model, api_key=...)

    def tools(self) -> list[BaseTool]:
        tool_llm = self.tool_llm()
        return [
            AskRecommendationTool(llm=tool_llm),
            AnalyzeFoodTool(llm=tool_llm),
            IngestContextTool(),
            LogMealTool(),
            RequestReportTool(),
        ]

    def metrics_collector(self) -> MetricsCollector: ...
    def orchestrator_service(self) -> OrchestratorService: ...
    def telegram_application(self) -> Application: ...
```

### MetricsCollector + MetricsCallbackHandler

`MetricsCallbackHandler` extends LangChain's `BaseCallbackHandler`:
- `on_llm_start` — records start time, model name
- `on_llm_end` — records token usage, computes latency and cost
- `on_tool_start` — records tool name, start time
- `on_tool_end` — records tool latency, success/failure

`MetricsCollector` holds request-scoped state:
- `start_request(request_id, telegram_id)` — initializes a new `RequestMetrics`
- Callback handler pushes metrics into the collector
- `end_request() -> RequestMetrics` — finalizes timing, logs the summary as structured JSON

**Cost table:** A dict mapping model names to `(input_cost_per_1k, output_cost_per_1k)`. Updated manually.

### Structured Logging

Python's `logging` module with a JSON formatter. All entries include:
- `request_id` for correlation
- `timestamp`
- `level`
- `message`
- Structured fields (metrics, errors, etc.)

### Entrypoint — `__main__.py`

```python
if __name__ == "__main__":
    settings = Settings.from_env()
    container = Container(settings)
    app = container.telegram_application()
    app.run_polling()
```

## Dependencies

```toml
dependencies = [
    "python-telegram-bot>=21.0",
    "langchain-core>=0.3",
    "langchain-openai>=0.3",
    "python-dotenv>=1.0",
]
```

## Testing Strategy

- **Domain tests:** Pure unit tests, no mocks needed for most (pure functions). Tool logic tests verify correct output for given inputs.
- **Port tests:** Verify `MessagingPort` ABC enforces implementation.
- **Service tests:** Mock `BaseChatModel` and tools to test orchestrator loop logic, error handling, max iterations.
- **Adapter tests:** Mock `python-telegram-bot` objects and `BaseChatModel` to test message handling, tool wrapping.
- **Infrastructure tests:** Monkeypatch env vars for Settings, verify Container wires correctly.
- **Metrics tests:** Verify callback handler captures correct metrics, collector aggregates properly.

No integration tests hitting real APIs in Phase 1. All external calls are mocked.

## Out of Scope (Phase 2+)

- Image/audio/PDF message handling
- Database persistence (PostgreSQL, pgvector)
- RAG context retrieval
- User profile and goal management
- Real meal logging
- Report generation
