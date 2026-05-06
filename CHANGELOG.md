# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **LangGraph agent pipeline**: Migrated from flat `create_agent` to a 6-node LangGraph `StateGraph` with explicit routing and conditional edges
- **Input guardrail**: LLM-based screening (gpt-4o-mini) blocks prompt injection, data extraction attempts, and manipulation before the agent executes
- **Output guardrail**: LLM-based validation (gpt-4o-mini) catches dangerous medical advice, eating disorder triggers, missing disclaimers, and data leaks
- **Response formatter**: Dedicated gpt-4o-mini node separating reasoning (agent) from communication (formatter) for independent prompt tuning
- **Memory management**: Automatic conversation history summarization when messages exceed configurable threshold, preventing context window bloat
- **Pinecone adapter**: New `PineconeEmbeddingRepository` implementing `EmbeddingRepository` port — zero changes to domain or services (hexagonal architecture port swap)
- **Daily nutrition summary**: Real `RequestReportTool` replacing stub — aggregates today's meals, compares against goals
- **LangGraph Platform deployment**: `langgraph.json` configuration for deploying to LangGraph Platform (free Developer tier)

### Changed
- **Orchestrator**: Simplified from monolith to thin graph coordinator
- **System prompt**: Split into `AGENT_SYSTEM_PROMPT` (reasoning + behavioral rules) and `FORMATTER_PROMPT` (tone + presentation)
- **Container**: Conditional Pinecone/pgvector adapter selection based on `PINECONE_API_KEY` env var

## [0.4.0] - 2026-04-13

### Added
- **Food image analysis**: Send a food photo and get a detailed nutritional breakdown (calories, protein, carbs, fat, fiber, sodium, sugar, saturated fat, cholesterol) via OpenAI Vision
- **Meal logging**: Log meals with full nutritional details from images or text descriptions, with optional timestamp for retroactive logging
- **Intent-based flow**: LLM decides whether to analyze-only or analyze+log based on user intent
- **Portion confirmation**: Kume estimates portion size and presents it for user confirmation before logging
- **Conversation history**: In-memory session history with 1-hour silence gap detection for multi-turn flows
- **VisionPort**: New output port abstracting vision AI providers, with OpenAI adapter
- **MealRepository**: New repository port + PostgreSQL adapter for meal persistence
- **SessionStore**: In-memory per-user conversation store with async-safe locking and stale user eviction
- **ImageStore**: Request-scoped image byte storage with MIME type tracking and contextvar-based request ID
- **Reasoning chain logging**: Dev console shows full agent reasoning — user messages, tool calls, tool results, and Kume's response
- **Alembic migration 002**: `meals` table with all nutritional columns
- **webp image support**: Added `image/webp` to ingestion service

### Changed
- **ContextBuilder**: Each section independently error-isolated; one failing data source doesn't prevent others from loading
- **ContextBuilder**: Recent meals section added (with timestamps) between lab results and current question
- **ContextBuilder**: Lab markers bounded to last 6 months to prevent unbounded context growth
- **ImageProcessor**: Updated from "coming soon" stub to neutral signal text
- **System prompt**: Added analyze_food_image and log_meal tool docs, intent-based logging rules, portion confirmation guidance
- **Orchestrator**: Session history loaded/saved per request, image bytes managed via ImageStore, RequestContext cleared in finally block
- **Orchestrator**: User message labeled as `[User message]` to match prompt's language detection instructions
- **Orchestrator**: Images labeled separately from documents in prompt to prevent index mismatch
- **Orchestrator**: Session history saves compact summaries (not full resource transcripts) to prevent context bloat
- **Metrics**: Request header shows user name when available
- **Settings**: Added `VISION_MODEL` env var (defaults to gpt-4o)
- **LogMealTool**: Replaced "coming soon" stub with real implementation; input validation clamps nutritional values at boundaries; invalid timestamps return error instead of silent fallback

### Fixed
- **Critical**: AnalyzeFoodImageTool request_id was never populated — image analysis always returned "not found"
- **Critical**: RequestContext never cleared after requests — stale user identity could leak across requests
- **Critical**: Image MIME type was discarded and hardcoded to JPEG — PNG/webp sent with wrong type to OpenAI
- **Critical**: Session history replayed full resource transcripts on subsequent turns, causing context bloat
- **Critical**: SessionStore race condition on overlapping requests for same user — added per-user async locking
- OpenAI Vision adapter assumed response.choices always has entries — added empty guard
- Log meal timestamp `.replace(tzinfo=UTC)` overwrote timezone instead of converting — now uses `.astimezone(UTC)`
- Stale user eviction now skips users with active locks to prevent mid-request eviction

## [0.3.0] - 2026-04-12

### Added
- Message debouncing and batching (3s silence window)
- Structured orchestrator input with Resource dataclass
- Onboarding UX and conversational system prompt
- Localized status messages (en/es) for media processing

## [0.2.0] - 2026-04-11

### Added
- Context ingestion pipeline (PDF, audio, text)
- PostgreSQL + pgvector for structured data and vector embeddings
- Repository pattern with focused ABCs (User, Goal, Restriction, Document, LabMarker)
- LLMPort abstraction with LangChain adapter
- Save tools (goals, restrictions, lab reports, health context)
- RAG context building with prescribed section order
- Resource processors (PDF via PyMuPDF, Audio via Whisper, Image stub)
- Alembic schema migrations
- Pretty dev logging formatter

## [0.1.0] - 2026-04-11

### Added
- Telegram bot with text message handling
- Agentic tool-use loop via LangChain
- Nutrition recommendation and food analysis tools
- Per-request metrics (tokens, cost, latency)
- Structured JSON logging
- Dual-model setup (orchestrator + tool model)
- Telegram HTML formatting
