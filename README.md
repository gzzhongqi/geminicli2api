# Gemini CLI to API

Access Google Gemini models using [Gemini CLI](https://github.com/google-gemini/gemini-cli) authentication. No API key required - just sign in with your Google account.

## Features

- **Free Access** - Uses Gemini CLI OAuth, no paid API key needed
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI chat completions
- **OpenAI Responses API** - Support for the newer `/v1/responses` endpoint
- **Anthropic-Compatible API** - Claude-style messages API
- **Native Gemini API** - Direct proxy to Google's Gemini API
- **Function Calling** - Full tool/function calling support across all API formats
- **Streaming Support** - Real-time streaming for all API formats
- **Multimodal** - Text and image inputs
- **Google Search Grounding** - Enable with `web_search` tool
- **Thinking Control** - Control reasoning via `reasoning_effort` parameter
- **Multiple Credentials** - Round-robin load balancing with automatic fallback

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run server
uv run python run.py
```

## Docker

```bash
# Using docker-compose
docker-compose up -d

# Or build manually
docker build -t geminicli2api .
docker run -p 8888:8888 \
  -e GEMINI_AUTH_PASSWORD=your_password \
  -e GOOGLE_APPLICATION_CREDENTIALS=oauth_creds.json \
  geminicli2api
```

## Deploy to Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/Nigp84?referralCode=xsbY2R)

Or deploy manually:

1. Fork this repository
2. Create a new project on [Railway](https://railway.app)
3. Connect your GitHub repo
4. Add environment variables:
   - `GEMINI_AUTH_PASSWORD` - Your API password
   - `GEMINI_CREDENTIALS_1` - First credential JSON (from `uv run python -m src.cli auth export`)
   - `GEMINI_CREDENTIALS_2` - (Optional) Additional credentials
5. Deploy!

## Credential Management CLI

Easily manage multiple Google accounts for load balancing:

```bash
# Add a new credential (opens browser for OAuth)
uv run python -m src.cli auth add
uv run python -m src.cli auth add --name "work-account"

# List all credentials
uv run python -m src.cli auth list

# Remove a credential
uv run python -m src.cli auth remove credential_1

# Export for deployment
uv run python -m src.cli auth export              # Print to stdout
uv run python -m src.cli auth export -o .env      # Save to file
uv run python -m src.cli auth export --docker     # Docker-compose format
```

Credentials are stored in `~/.geminicli2api/credentials/`.

## Configuration

### Single Credential (Default)

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_AUTH_PASSWORD` | No | Password for API access (if empty, no auth required) |
| `GOOGLE_APPLICATION_CREDENTIALS` | No* | Path to OAuth credentials file |
| `GEMINI_CREDENTIALS` | No* | OAuth credentials as JSON string |

*If no credentials provided, browser OAuth flow will start on first run

### Multiple Credentials (Load Balancing)

For high availability and load distribution, you can configure multiple Gemini CLI credentials:

```bash
# Option 1: Indexed environment variables
GEMINI_CREDENTIALS_1='{"refresh_token":"xxx","client_id":"...","client_secret":"..."}'
GEMINI_CREDENTIALS_2='{"refresh_token":"yyy","client_id":"...","client_secret":"..."}'
GEMINI_CREDENTIALS_3='{"refresh_token":"zzz","client_id":"...","client_secret":"..."}'

# Option 2: Multiple credential files
GEMINI_CREDENTIAL_FILES=creds1.json,creds2.json,creds3.json
```

**How it works:**
- **Round-robin selection** - Requests are distributed evenly across all credentials
- **Automatic fallback** - If a credential fails (401, 403, 429), automatically retry with another
- **Recovery time** - Failed credentials are retried after 5 minutes
- **Backward compatible** - Single credential setup continues to work

**Workflow for multiple accounts:**
1. Use `uv run python -m src.cli auth add` to authenticate each Google account
2. Export with `uv run python -m src.cli auth export -o .env`
3. Copy the `.env` file to your server or use with docker-compose

## API Endpoints

### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/responses` - Responses API
- `GET /v1/models` - List models

### Anthropic-Compatible
- `POST /v1/messages` - Messages API

### Native Gemini
- `GET /v1beta/models` - List models
- `POST /v1beta/models/{model}:generateContent` - Generate content
- `POST /v1beta/models/{model}:streamGenerateContent` - Stream content

### Utility
- `GET /health` - Health check

## Authentication

On first run, the server will open a browser for Google OAuth login. Credentials are saved to `oauth_creds.json` for subsequent runs.

Alternatively, set credentials via environment variables:
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to OAuth credentials file
- `GEMINI_CREDENTIALS` - OAuth credentials as JSON string

API authentication supports:
- Bearer Token: `Authorization: Bearer <password>`
- Basic Auth: `Authorization: Basic <base64>`
- Query Parameter: `?key=<password>`
- Google Header: `x-goog-api-key: <password>`

## Usage Example

### OpenAI Chat Completions

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="your_password"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Function Calling

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="your_password"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Responses API

```python
import httpx

response = httpx.post(
    "http://localhost:8888/v1/responses",
    headers={"Authorization": "Bearer your_password"},
    json={
        "model": "gemini-2.5-flash",
        "input": "What is 2+2?",
    }
)
print(response.json()["output_text"])
```

### Responses API with Function Calling

```python
import httpx

response = httpx.post(
    "http://localhost:8888/v1/responses",
    headers={"Authorization": "Bearer your_password"},
    json={
        "model": "gemini-2.5-flash",
        "input": "What's the weather in Paris?",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        ]
    }
)

# Check for function calls in output
for item in response.json()["output"]:
    if item["type"] == "function_call":
        print(f"Call ID: {item['call_id']}")
        print(f"Function: {item['name']}")
        print(f"Arguments: {item['arguments']}")
```

### Responses API with Web Search

```python
import httpx

response = httpx.post(
    "http://localhost:8888/v1/responses",
    headers={"Authorization": "Bearer your_password"},
    json={
        "model": "gemini-2.5-flash",
        "input": "What are the latest news about AI?",
        "tools": [{"type": "web_search"}]
    }
)
print(response.json()["output_text"])
```

## Models

Supported models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`

## Parameters

### Reasoning/Thinking Control

Control thinking budget via `reasoning_effort` parameter (OpenAI) or `thinking` config (Anthropic):

```python
# OpenAI format
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Solve this math problem"}],
    reasoning_effort="high"  # none, off, minimal, low, medium, high, max
)

# Anthropic format
response = client.messages.create(
    model="gemini-2.5-flash",
    max_tokens=1024,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Solve this math problem"}]
)
```

| Value | Description | Flash Budget | Pro Budget |
|-------|-------------|--------------|------------|
| `none`/`off`/`disabled` | Disable thinking | 0 | 128 |
| `minimal` | Minimal thinking | 0 | 128 |
| `low` | Low thinking | 1000 | 1000 |
| `medium` | Auto (default) | -1 | -1 |
| `high` | High thinking | 24576 | 32768 |
| `max` | Maximum thinking | 24576 | 32768 |

### Web Search / Grounding

Enable Google Search grounding via `web_search` tool:

```python
# OpenAI format
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "What are the latest AI news?"}],
    tools=[{"type": "web_search"}]
)

# Responses API format
response = httpx.post(
    "http://localhost:8888/v1/responses",
    json={
        "model": "gemini-2.5-flash",
        "input": "What are the latest AI news?",
        "tools": [{"type": "web_search"}]
    }
)
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.
