# Bootstrap Skill — Design Spec

## Overview

A skill that scaffolds a new project from a single command. Takes a folder, a tech stack, an architecture pattern, an optional Notion project name, and an optional GitHub URL — and produces a fully configured repo with CLAUDE.md, CI/CD, pre-commit hooks, folder structure, an open PR, and Notion tasks from the PRD.

**Invocation example:**
```
Bootstrap this folder with Python, UV, hexagonal architecture.
Notion project is called Kume Agent. GitHub: https://github.com/user/kume-agent
```

## Skill Structure

Approach B — lean `SKILL.md` for orchestration (~200-300 lines) with bundled references:

```
bootstrap/
├── SKILL.md                          # Orchestration flow
└── references/
    ├── claude-md-template.md         # Fixed-zone CLAUDE.md template
    ├── python-uv/
    │   ├── pyproject.toml.md         # Template with ruff, pytest config
    │   ├── pre-commit-config.yaml.md
    │   ├── lint.yml.md
    │   ├── tests.yml.md
    │   ├── reviewdog.yml.md
    │   └── gitignore.md
    └── typescript-node/
        ├── package.json.md
        ├── tsconfig.json.md
        ├── pre-commit-config.yaml.md
        ├── lint.yml.md
        ├── tests.yml.md
        ├── reviewdog.yml.md
        └── gitignore.md
```

## Inputs

Parsed from user message:

| Input | Required | Example |
|-------|----------|---------|
| Technologies (language, pkg manager, framework) | Yes | "Python, UV, FastAPI" |
| Architecture pattern | Yes | "hexagonal architecture" |
| Notion project name | No | "Kume Agent" |
| GitHub URL | No | "https://github.com/user/kume-agent" |

PRD is auto-detected by scanning for `*prd*`, `*PRD*`, or `.md` files with product/requirements content in the project root.

For any missing non-required info (linter, test framework, Python version), the skill either proposes sensible defaults or asks the user.

## Orchestration Flow

```
1. DISCOVER — Scan project folder
   ├── Find PRD or similar docs
   ├── Check if CLAUDE.md already exists
   ├── Check if git remote is set
   └── Check current project structure

2. NOTION SETUP — Connect to Notion (if project name provided)
   ├── Invoke notion-tasks skill to search for project by name
   ├── Retrieve: project page ID, tasks database ID, data sources
   └── If not found: ask user to confirm project name or provide IDs

3. GENERATE CLAUDE.md
   ├── Fixed sections from template: Notion, Skills, Code Review, Conventions
   │   └── Conventions adapted to tech stack (Python/UV/ruff or TS/pnpm/eslint etc.)
   ├── Dynamic sections generated from PRD + tech inputs:
   │   ├── Architecture (based on pattern + PRD analysis)
   │   └── Critical Rules (from PRD constraints + domain concerns)
   └── Write to project root

4. SCAFFOLD — Create project structure
   ├── Folder skeleton based on architecture pattern + language
   ├── Config files (pyproject.toml / package.json, .gitignore, etc.)
   ├── Empty __init__.py / index files
   ├── .github/workflows/ (lint, tests, reviewdog)
   ├── .pre-commit-config.yaml
   └── Move PRD to docs/ folder

5. GIT + PR
   ├── If GitHub URL provided and no remote: git remote add origin
   ├── First commit on main (if no commits yet)
   ├── Create branch: chore/bootstrap-project
   └── Raise PR using tars skill

6. NOTION TASKS — Break down PRD into phases (if PRD exists)
   ├── Read and analyze PRD
   ├── Decompose into linear, self-contained phases
   ├── Create one Notion task per phase in "Planning" status
   ├── Each task notes: "Use superpowers skill to spec/plan"
   └── Phases ordered but each independently functional
```

Steps 5 and 6 are independent — the PR is for scaffolding, the tasks are for PRD phases.

## CLAUDE.md Structure

### Fixed Zone (template with slot values)

```markdown
# Project Configuration

## Notion
- **Project name**: {{project_name}}
- **Project page**: {{project_page_id}}
- **Tasks database**: {{tasks_database_id}}
- **Tasks data source**: {{tasks_data_source}}
- **Projects data source**: {{projects_data_source}}

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
- Package manager: `{{package_manager}}`
- {{language}} version: {{version}}
- Linter: `{{linter}}`
- Tests: `{{test_framework}}` (run with `{{test_command}}`)
- Coverage: `{{coverage_command}}`
- Branch naming: `feat/`, `fix/`, `chore/`
- Commits end with `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
```

### Dynamic Zone (generated from PRD + architecture)

```markdown
## Architecture
{{Generated from architecture pattern + PRD. For hexagonal: describes
  ports/, adapters/, services/, domain/ and their responsibilities.
  Includes data flow if PRD has enough detail.}}

## Critical Rules
{{Extracted from PRD constraints and domain concerns.}}
```

If no PRD exists, dynamic zone is minimal — architecture pattern description + placeholder Critical Rules.

## Scaffolding Patterns

### Hexagonal Architecture (Python)

```
project_root/
├── src/
│   └── {{project_slug}}/
│       ├── __init__.py
│       ├── ports/
│       │   └── __init__.py
│       ├── adapters/
│       │   └── __init__.py
│       ├── services/
│       │   └── __init__.py
│       └── domain/
│           └── __init__.py
├── tests/
│   └── __init__.py
├── docs/
│   └── {{prd_filename}}
├── .github/
│   └── workflows/
│       ├── lint.yml
│       ├── tests.yml
│       └── reviewdog.yml
├── .pre-commit-config.yaml
├── CLAUDE.md
├── README.md
├── .gitignore
└── pyproject.toml
```

### Python/UV CI Workflows

**lint.yml** — Triggers on push to main + PRs, path-filtered to `src/**`, `tests/**`, `pyproject.toml`:
- Setup: `actions/setup-python@v5` (3.11), `astral-sh/setup-uv@v4`, `uv sync --extra dev`
- Jobs: `uv run ruff check src/ tests/` + `uv run ruff format --check src/ tests/`

**tests.yml** — Same triggers and setup:
- `uv run pytest tests/ -v --tb=short --junitxml=test-results.xml`
- `mikepenz/action-junit-report@v5` to post results on PRs

**reviewdog.yml** — PR-only, `permissions: contents read, pull-requests write`:
- **Ruff job:** `reviewdog/action-setup@v1`, then:
  ```
  uv run ruff check --output-format=rdjson src/ tests/ |
    reviewdog -f=rdjson -name="ruff" -reporter=github-pr-review \
      -filter-mode=added -level=error -fail-level=error
  ```
- **mypy job:**
  ```
  uv run mypy src/ --no-error-summary 2>&1 |
    reviewdog -efm="%f:%l: %t%*[a-z]: %m" -name="mypy" \
      -reporter=github-pr-review -filter-mode=added -fail-level=error
  ```

### Python Pre-commit

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

### Python pyproject.toml Config

```toml
[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501", "SIM108", "SIM105", "SIM102", "UP042"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### TypeScript/Node CI Workflows

**lint.yml** — Triggers on push to main + PRs, path-filtered:
- Setup: `actions/setup-node@v4` (22), `npm ci`
- Jobs: `npx eslint src/` + `npx prettier --check "src/**/*.{ts,tsx,js,css}"`

**tests.yml** — Same setup:
- `npx tsc --noEmit` + `npx jest --ci --verbose`

**reviewdog.yml** — PR-only:
- **ESLint job:** `reviewdog/action-eslint@v1`, `reporter: github-pr-review`, `filter_mode: added`, `fail_level: error`
- **Prettier job:** `EPMatt/reviewdog-action-prettier@v1.3.0`, same config

### TypeScript Pre-commit

```yaml
repos:
  - repo: local
    hooks:
      - id: eslint
        name: eslint
        entry: npx eslint --fix
        language: system
        files: \.(ts|tsx|js|jsx)$
      - id: prettier
        name: prettier
        entry: npx prettier --write --ignore-unknown
        language: system
        files: \.(ts|tsx|js|jsx|css)$
```

### What scaffolding does NOT generate
- No application code (no stubs, no implementations)
- No Docker files (separate task)
- No release/deploy workflows (separate task)

## PRD Phase Breakdown

When a PRD is found:

1. **Read and analyze** — identify phases/milestones/roadmap sections
2. **Decompose into self-contained phases** — each produces a working system independently, numbered linearly
3. **Create one Notion task per phase** in "Planning" status via `notion-tasks`

### Task format
- **Title:** `Phase N: {{phase_name}}`
- **Description:** Summary of what this phase delivers, from the PRD
- **Status:** Planning
- **Note:** "Use `superpowers` skill to create a detailed spec and implementation plan before coding"

### Rules
- Phases ordered linearly but each independently functional
- No sub-tasks within phases — that happens during `superpowers:writing-plans`
- If PRD has no clear phases, propose breakdown to user for approval before creating tasks
- PRD moved to `docs/{{prd_filename}}`

## Skills Used

| Skill | When | Purpose |
|-------|------|---------|
| `notion-tasks` | Step 2, Step 6 | Discover project IDs, create phase tasks |
| `tars` | Step 5 | Push code, create PR via bot identity |

## Other Tech Stacks

Python/UV and TypeScript/Node have baked-in templates for CI, pre-commit, and config files. For any other stack (Go, Rust, Java, etc.), the skill:
1. Proposes standard tooling for that ecosystem (linter, test runner, formatter)
2. Asks the user to confirm before generating
3. Generates CI workflows and pre-commit config following the same patterns (reviewdog for PR review, separate lint + test workflows)

No baked-in templates for other stacks — the skill reasons about them dynamically.

## Not in Scope

- Application code generation
- Docker / containerization
- CI/CD for deployment (release workflows)
- Database setup
- Environment variable configuration
