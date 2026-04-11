# PRD — Kume (Hexagonal Architecture)

## 1. Project Name

**Kume**

> Kume is a multimodal AI agent that structures fragmented personal health context and food inputs into actionable nutrition guidance via Telegram.

---

## 2. Vision

Kume is a **personal AI agent** that:
- ingests health-related data (PDFs, images, audio, text)
- builds structured and semantic user context
- answers food-related questions grounded in that context
- tracks user behavior over time
- generates reports for self-tracking and professionals

The system emphasizes:
- **RAG (Retrieval-Augmented Generation)**
- **multimodal ingestion**
- **agent/tool orchestration**
- **structured + semantic memory separation**
- **production-like backend architecture**

---

## 3. Product Goal (Open Source)

This project is designed to demonstrate:

- real-world applied AI
- RAG with user-specific context
- multimodal pipelines (text, audio, image, PDF)
- agent orchestration with tools
- structured data modeling
- report generation
- clean architecture (Hexagonal)

---

## 4. Problem Statement

Users dealing with nutrition and health:
- have fragmented data (labs, diets, habits, meals)
- lack a unified interface
- struggle to connect daily decisions with long-term goals

Kume solves this by acting as a **context-aware conversational system**.

---

## 5. Core Capabilities

### 5.1 Context Ingestion
- PDFs (medical reports, diets)
- audio (goals, explanations)
- text
- images

### 5.2 Recommendation / Analysis
- “Can I eat this?” + image
- “What should I eat today?”
- “Is this aligned with my goals?”

### 5.3 Tracking & Reporting
- meal logging
- historical analysis
- exportable reports (CSV/XLSX)

---

## 6. Architecture — Hexagonal (Ports & Adapters)

Kume follows **Hexagonal Architecture (Ports & Adapters)** to:
- isolate domain logic
- make tools replaceable
- support multiple interfaces (Telegram today, others tomorrow)

---

## 7. High-Level Architecture

### Core Layers

```
           ┌───────────────────────────────┐
           │        External World         │
           │ Telegram / OpenAI / Storage   │
           └─────────────┬─────────────────┘
                         │
        ┌────────────────▼─────────────────┐
        │         Adapters Layer           │
        │ (Telegram, OpenAI, DB, Files)    │
        └────────────────┬─────────────────┘
                         │
        ┌────────────────▼─────────────────┐
        │            Ports Layer           │
        │ Interfaces (contracts)           │
        └────────────────┬─────────────────┘
                         │
        ┌────────────────▼─────────────────┐
        │           Domain Core            │
        │ Business logic & entities        │
        └──────────────────────────────────┘
```

---

## 8. Domain Core

### Responsibilities
- user context modeling
- decision logic (what to retrieve, how to respond)
- intent classification
- orchestration rules

### Entities

- `User`
- `Goal`
- `Restriction`
- `Document`
- `Meal`
- `LabMarker`
- `ConversationEvent`

### Value Objects

- `NutritionAssessment`
- `ConfidenceScore`
- `ContextBundle`

---

## 9. Ports (Interfaces)

Ports define what the domain needs, not how it’s implemented.

### Input Ports (Driving)

- `ReceiveUserMessage`
- `ProcessContextInput`
- `AnalyzeFoodInput`
- `GenerateReport`

### Output Ports (Driven)

- `LLMPort`
- `EmbeddingPort`
- `VisionPort`
- `SpeechToTextPort`
- `VectorStorePort`
- `DatabasePort`
- `FileStoragePort`
- `MessagingPort`

---

## 10. Adapters

Adapters implement ports.

### Driving Adapters

- Telegram Bot Adapter
- REST API (optional)

### Driven Adapters

- OpenAI Adapter (LLM, embeddings, vision, STT)
- PostgreSQL Adapter
- pgvector Adapter
- File Storage Adapter (local/S3)

---

## 11. Agent Orchestration (Inside Domain)

Instead of a fully autonomous agent, Kume uses:

- intent classification
- controlled tool execution
- structured prompt building

### Intents

- `INGEST_CONTEXT`
- `ASK_FOOD_ANALYSIS`
- `ASK_RECOMMENDATION`
- `LOG_MEAL`
- `REQUEST_REPORT`

---

## 12. RAG Strategy

### Indexed Data

- document chunks
- summaries
- user goals
- restrictions

### Retrieval

- semantic search (top-k)
- filtered by type
- optional reranking

### Prompt Context

1. user profile
2. goals
3. restrictions
4. relevant documents
5. recent meals
6. current input

---

## 13. Data Model (Simplified)

### User
- id
- telegram_id

### Goal
- user_id
- description

### Restriction
- user_id
- type

### Document
- user_id
- type
- summary

### Meal
- user_id
- timestamp
- description
- assessment

### LabMarker
- document_id
- name
- value

---

## 14. Functional Requirements

- accept Telegram messages
- support text, audio, images, PDFs
- transcribe audio
- extract text from PDFs
- store structured + semantic context
- perform RAG-based responses
- analyze food images
- log meals
- generate reports

---

## 15. Non-Functional Requirements

- modular architecture
- replaceable AI providers
- local-first setup
- observability hooks
- cost control

---

## 16. Tech Stack (Recommended)

- Python + FastAPI
- python-telegram-bot
- PostgreSQL + pgvector
- OpenAI API
- SQLAlchemy
- Pandas / OpenPyXL

---

## 17. Key Design Decisions

1. Use Hexagonal Architecture for clarity and extensibility
2. Separate structured vs semantic memory
3. Avoid fully autonomous agents initially
4. Prefer tool orchestration + control

---

## 18. Roadmap

### Phase 1
- Telegram bot
- basic responses

### Phase 2
- context ingestion
- embeddings

### Phase 3
- food analysis (image)

### Phase 4
- meal logging

### Phase 5
- reporting

---

## 19. Repo Structure

```
src/
  domain/
  application/
  ports/
  adapters/
  infrastructure/
```

---

## 20. Positioning

Kume is not just a chatbot.

It is:

> A multimodal, context-aware AI agent built with clean architecture principles, designed to demonstrate real-world RAG, agent orchestration, and health data structuring.

---

## 21. Next Steps

- define MVP scope
- implement ports
- implement Telegram adapter
- implement ingestion pipeline
- add RAG
- add reporting

---

## 22. Final Note

Kume should feel like an **entity**, not a feature.

Everything in the system — naming, architecture, and behavior — should reinforce that.

