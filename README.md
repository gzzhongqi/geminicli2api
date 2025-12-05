# geminicli2api

Access Google Gemini models using [Gemini CLI](https://github.com/google-gemini/gemini-cli) authentication. No API key required - just sign in with your Google account.

## Features

- **Free Access** - Uses Gemini CLI OAuth, no paid API key needed
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI chat completions
- **Native Gemini API** - Direct proxy to Google's Gemini API
- **Streaming Support** - Real-time streaming for both API formats
- **Multimodal** - Text and image inputs
- **Google Search Grounding** - Enable with `-search` model suffix
- **Thinking Control** - `-nothinking` and `-maxthinking` model variants

## Quick Start

```bash
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

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_AUTH_PASSWORD` | No | Password for API access (default: `123456`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | No* | Path to OAuth credentials file |
| `GEMINI_CREDENTIALS` | No* | OAuth credentials as JSON string |

*If no credentials provided, browser OAuth flow will start on first run

## API Endpoints

### OpenAI-Compatible
- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models

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

## Models

Base models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`

Variants (2.5 models):
- `-search` - Google Search grounding
- `-nothinking` - Minimal reasoning
- `-maxthinking` - Maximum reasoning budget

## License

MIT License - see [LICENSE](LICENSE) for details.
