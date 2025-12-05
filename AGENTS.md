# AGENTS.md

## Build & Run
- Install: `uv sync`
- Run locally: `uv run python run.py` (port 8888)
- Docker: `docker-compose up -d` or `docker build -t geminicli2api . && docker run -p 8888:8888 geminicli2api`
- No test suite exists; no lint/format tooling configured

## Project Structure
- `src/`: Main package (config, routes, transformers, auth, models, utils)
- `run.py`: Entry point
- Config via environment variables (see `.env.example`)

## Code Style

### Formatting & Naming
- 4-space indent, double quotes for strings
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Private/helper functions prefixed with underscore (`_helper_function`)

### Imports
- Order: stdlib → third-party → local relative (`from .module import ...`)
- Remove unused imports

### Type Hints
- Required on all function parameters and return types
- Use `Optional[T]` for nullable types, `list[T]`/`dict[K, V]` for collections
- Use Pydantic models for request/response schemas

### Error Handling
- Use specific exceptions (no bare `except:`)
- Log with `logging` module
- Use `create_error_response()` from config for JSON errors
- Format: `{"error": {"message": ..., "type": ..., "code": ...}}`

### Documentation
- Module-level docstrings describing purpose
- Function docstrings for public/complex functions

### Clean Code Patterns
- **Extract helpers**: Break large functions into smaller, focused helpers
- **DRY**: Extract duplicated logic into reusable functions
- **Constants**: Define magic values in `config.py` (e.g., `STREAMING_RESPONSE_HEADERS`, `MODEL_CREATED_TIMESTAMP`)
- **Single responsibility**: Each function should do one thing well
- **Meaningful names**: Function names should describe what they do (e.g., `_parse_data_uri`, `_extract_content_and_reasoning`)

### FastAPI Patterns
- Use `lifespan` context manager instead of deprecated `@app.on_event`
- Keep route handlers thin; delegate to helper functions
- Streaming responses use `StreamingResponse` with async generators
