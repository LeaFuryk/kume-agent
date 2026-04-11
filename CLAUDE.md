# Project Configuration

## Notion
- **Project name**: Kume Agent
- Notion IDs are stored in `.env` to keep them out of version control.
- Copy values from `.env.example` and fill in your own Notion workspace IDs.
- Required variables: `NOTION_PROJECT_PAGE`, `NOTION_TASKS_DB`, `NOTION_TASKS_DS`, `NOTION_PROJECTS_DS`

## Skills
- `notion-tasks` — Task management from Notion board. Always check Notion before starting work.
- `tars` — All GitHub operations via tars-bot-01 GitHub App (push, PRs, comments, reviews)
- `codex` — Code review gate. Run `/codex:rescue --model gpt-5.4` after implementation to review changes.
- `superpowers` — Planning, TDD, systematic debugging, parallel agents, code review, git worktrees.

## Code Review
- After writing or modifying code, run a Codex review (`/codex:rescue --model gpt-5.4`) on changed files before marking work as complete
- Fix any issues found by the review before presenting results
- Run Codex repeatedly until it reports "No issues found"

## Conventions
- Package manager: `uv`
- Python version: 3.11+
- Linter: `ruff`
- Tests: `pytest` (run with `uv run pytest tests/ -v`)
- Coverage: `uv run pytest --cov --cov-report=term-missing`
- Branch naming: `feat/`, `fix/`, `chore/`
- Commits end with `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

---

## Architecture — Hexagonal (Ports & Adapters)

Kume follows **Hexagonal Architecture** to isolate domain logic from infrastructure concerns. Every external dependency is accessed through a port (interface) and implemented by an adapter.

### Layers

```
src/kume/
├── domain/          # Entities, value objects, domain logic — zero external imports
├── ports/           # Abstract interfaces (input + output)
│   ├── input/       # Driving ports: ReceiveUserMessage, ProcessContextInput, AnalyzeFoodInput, GenerateReport
│   └── output/      # Driven ports: LLMPort, EmbeddingPort, VisionPort, SpeechToTextPort,
│                    #   VectorStorePort, DatabasePort, FileStoragePort, MessagingPort
├── services/        # Application services — orchestrate domain + ports
├── adapters/        # Concrete implementations of ports
│   ├── input/       # Driving: TelegramAdapter, (future) RESTAdapter
│   └── output/      # Driven: OpenAIAdapter, PostgresAdapter, PgvectorAdapter, FileStorageAdapter
└── infrastructure/  # App bootstrap, config, dependency injection
```

### Data Flow

```
Telegram message
  → TelegramAdapter (input adapter)
    → ReceiveUserMessage (input port)
      → MessageService (application service)
        → IntentClassifier (domain logic)
        → [based on intent]:
            INGEST_CONTEXT  → ContextIngestionService → EmbeddingPort + VectorStorePort + DatabasePort
            ASK_FOOD_ANALYSIS → FoodAnalysisService → VisionPort + LLMPort + VectorStorePort
            ASK_RECOMMENDATION → RecommendationService → LLMPort + VectorStorePort + DatabasePort
            LOG_MEAL → MealLoggingService → DatabasePort
            REQUEST_REPORT → ReportService → DatabasePort + FileStoragePort
        → MessagingPort (output port)
      → TelegramAdapter sends response
```

### Domain Entities
- `User`, `Goal`, `Restriction`, `Document`, `Meal`, `LabMarker`, `ConversationEvent`

### Value Objects
- `NutritionAssessment`, `ConfidenceScore`, `ContextBundle`

---

## Critical Rules

1. **Domain purity**: `domain/` MUST have zero imports from `adapters/`, `infrastructure/`, or external libraries. Domain logic depends only on Python stdlib and domain types.
2. **Port contracts**: Adapters MUST implement their port's abstract interface. Never call an adapter directly from a service — always go through the port.
3. **No dict soup**: Use typed models (dataclasses or Pydantic) for all data crossing boundaries. Raw dicts are forbidden inside `domain/` and `services/`.
4. **Dependency direction**: Dependencies flow inward only: `adapters → ports ← services → domain`. Never import outward.
5. **One adapter per port**: Each port has exactly one active adapter at runtime, injected via `infrastructure/`. Swapping providers (e.g. OpenAI → Anthropic) means writing a new adapter, not modifying an existing one.
6. **User data isolation**: All queries MUST be scoped to the authenticated `user_id`. Never return or modify another user's data.
7. **RAG context building**: Prompt context follows this order: user profile → goals → restrictions → relevant documents → recent meals → current input. Do not reorder.
8. **Intent-driven orchestration**: The domain classifies intent first, then dispatches to the correct service. No service should handle multiple intents.
