---
title: Gemini CLI to API Proxy
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Gemini CLI to API Proxy (geminicli2api)

A FastAPI-based proxy server that converts the Gemini CLI tool into both OpenAI-compatible and native Gemini API endpoints. This allows you to leverage Google's free Gemini API quota through familiar OpenAI API interfaces or direct Gemini API calls.

## 🚀 Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions API
- **Native Gemini API**: Direct proxy to Google's Gemini API
- **Streaming Support**: Real-time streaming responses for both API formats
- **Multimodal Support**: Text and image inputs
- **Authentication**: Multiple auth methods (Bearer, Basic, API key)
- **Google Search Grounding**: Enable Google Search for grounded responses using `-search` models.
- **Thinking/Reasoning Control**: Control Gemini's thinking process with `-nothinking` and `-maxthinking` models.
- **Docker Ready**: Containerized for easy deployment
- **Hugging Face Spaces**: Ready for deployment on Hugging Face

## 🔧 Environment Variables

### Required
- `GEMINI_AUTH_PASSWORD`: Authentication password for API access

### Optional Credential Sources (choose one)
- `GEMINI_CREDENTIALS`: JSON string containing Google OAuth credentials
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google OAuth credentials file
- `GOOGLE_CLOUD_PROJECT`: Google Cloud project ID
- `GEMINI_PROJECT_ID`: Alternative project ID variable

### Optional Robustness / Retry Tuning
- `UPSTREAM_CONNECT_TIMEOUT_S` (default: `10`, set to `0` or `None` for no timeout)
- `UPSTREAM_READ_TIMEOUT_S` (default: `0` (None), set to `0` or `None` for no timeout)
- `UPSTREAM_STREAM_READ_TIMEOUT_S` (default: `0` (None), set to `0` or `None` for no timeout)
- `UPSTREAM_MAX_ATTEMPTS` (default: `4`)
- `UPSTREAM_BACKOFF_BASE_S` (default: `0.5`)
- `UPSTREAM_BACKOFF_MAX_S` (default: `8`)
- `UPSTREAM_MAX_CONNECTIONS` (default: `100`)
- `UPSTREAM_MAX_KEEPALIVE_CONNECTIONS` (default: `20`)
- `ONBOARD_POLL_INTERVAL_S` (default: `2.5`)
- `ONBOARD_MAX_WAIT_S` (default: `90`)

### Example Credentials JSON
```json
{
  "client_id": "your-client-id",
  "client_secret": "your-client-secret", 
  "token": "your-access-token",
  "refresh_token": "your-refresh-token",
  "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

## 📡 API Endpoints

### OpenAI-Compatible Endpoints
- `POST /v1/chat/completions` - Chat completions (streaming & non-streaming)
- `GET /v1/models` - List available models

### Native Gemini Endpoints  
- `GET /v1beta/models` - List Gemini models
- `POST /v1beta/models/{model}:generateContent` - Generate content
- `POST /v1beta/models/{model}:streamGenerateContent` - Stream content
- All other Gemini API endpoints are proxied through

### Utility Endpoints
- `GET /health` - Health check for container orchestration

## 🔐 Authentication

The API supports multiple authentication methods:

1. **Bearer Token**: `Authorization: Bearer YOUR_PASSWORD`
2. **Basic Auth**: `Authorization: Basic base64(username:YOUR_PASSWORD)`
3. **Query Parameter**: `?key=YOUR_PASSWORD`
4. **Google Header**: `x-goog-api-key: YOUR_PASSWORD`

## 🐳 Docker Usage

```bash
# Build the image
docker build -t geminicli2api .

# Run on default port 8888 (compatibility)
docker run -p 8888:8888 \
  -e GEMINI_AUTH_PASSWORD=your_password \
  -e GEMINI_CREDENTIALS='{"client_id":"...","token":"..."}' \
  -e PORT=8888 \
  geminicli2api

# Run on port 7860 (Hugging Face compatible)
docker run -p 7860:7860 \
  -e GEMINI_AUTH_PASSWORD=your_password \
  -e GEMINI_CREDENTIALS='{"client_id":"...","token":"..."}' \
  -e PORT=7860 \
  geminicli2api
```

### Docker Compose

```bash
# Default setup (port 8888)
docker-compose up -d

# Hugging Face setup (port 7860)
docker-compose --profile hf up -d geminicli2api-hf
```

## 🤗 Hugging Face Spaces

This project is configured for Hugging Face Spaces deployment:

1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your repository
4. Set the required environment variables in Space settings:
   - `GEMINI_AUTH_PASSWORD`
   - `GEMINI_CREDENTIALS` (or other credential source)

The Space will automatically build and deploy using the included Dockerfile.

## 📝 OpenAI API Example

```python
import openai

# Configure client to use your proxy
client = openai.OpenAI(
    base_url="http://localhost:8888/v1",  # or 7860 for HF
    api_key="your_password"  # Your GEMINI_AUTH_PASSWORD
)

# Use like normal OpenAI API
response = client.chat.completions.create(
    model="gemini-2.5-pro-maxthinking",
    messages=[
        {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ],
    stream=True
)

# Separate reasoning from the final answer
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(f"Thinking: {chunk.choices[0].delta.reasoning_content}")
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 🔧 Native Gemini API Example

```python
import requests

headers = {
    "Authorization": "Bearer your_password",
    "Content-Type": "application/json"
}

data = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Explain the theory of relativity in simple terms."}]
        }
    ],
    "thinkingConfig": {
        "thinkingBudget": 32768,
        "includeThoughts": True
    }
}

response = requests.post(
    "http://localhost:8888/v1beta/models/gemini-2.5-pro:generateContent",  # or 7860 for HF
    headers=headers,
    json=data
)

print(response.json())
```

## 🎯 Supported Models

### Base Models
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-1.0-pro`

### Model Variants
The proxy automatically creates variants for `gemini-2.5-pro` and `gemini-2.5-flash` models:

- **`-search`**: Appends `-search` to a model name to enable Google Search grounding.
  - Example: `gemini-2.5-pro-search`
- **`-nothinking`**: Appends `-nothinking` to minimize reasoning steps.
  - Example: `gemini-2.5-flash-nothinking`
- **`-maxthinking`**: Appends `-maxthinking` to maximize the reasoning budget.
  - Example: `gemini-2.5-pro-maxthinking`

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
