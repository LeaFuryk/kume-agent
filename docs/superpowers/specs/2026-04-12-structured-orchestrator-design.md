# Structured Orchestrator Input — Design Spec

## Overview

Replace the combined text blob interface with structured input: user_message, user_name, resources, and resource_transcripts. The system prompt includes a resource-to-tool mapping with concrete examples for every tool.

## Changes

### 1. Resource dataclass

```python
@dataclass
class Resource:
    mime_type: str
    transcript: str
    raw_bytes: bytes | None  # kept for image tools
```

Lives in `services/orchestrator.py` or its own file.

### 2. OrchestratorService.process() new signature

```python
async def process(
    self,
    telegram_id: int,
    user_message: str,
    user_name: str | None = None,
    resources: list[Resource] | None = None,
) -> str:
```

Builds the HumanMessage dynamically:
- User prefix (name)
- User message
- Resource section with type counts + transcripts
- Instruction to use appropriate tool per resource type

### 3. System prompt additions (prompts.py)

Resource-to-tool mapping:
- PDF → process_lab_report(texts=[...])
- Food images → analyze_food (coming soon)
- Audio → already transcribed, treat as text

Tool examples (English):
- process_lab_report: "User sends 3 PDF attachments → process_lab_report(texts=[t1, t2, t3])"
- save_goal: "User says 'I want to lower my triglycerides' → save_goal(...)"
- save_restriction: "User says 'I'm lactose intolerant' → save_restriction(...)"
- save_health_context: "User says 'I weigh 80kg' → save_health_context(...)"
- fetch_user_context: "User asks 'what were my results?' → fetch_user_context(...)"
- ask_recommendation: "User asks 'what should I eat?' → ask_recommendation(...)"
- analyze_food: "User asks 'can I eat pizza?' → analyze_food(...)"

### 4. Batch handler changes

`_process_batch` collects resources separately and passes structured data to orchestrator instead of combining into one text blob.

## Files changed

- `src/kume/services/orchestrator.py` — new process() signature, Resource dataclass, dynamic prompt building
- `src/kume/services/prompts.py` — tool mapping, examples
- `src/kume/adapters/input/telegram_bot.py` — _process_batch passes resources separately
- Tests updated accordingly
