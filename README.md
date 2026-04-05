# Accountant Chatbot Backend

A production-ready agentic chatbot with LLM integration, RAG capabilities, and extensible architecture.

## Features

- **Multi-LLM Support**: OpenAI (GPT-4) and Anthropic (Claude)
- **Agentic Capabilities**: Tool calling with iterative reasoning
- **RAG Pipeline**: Document ingestion, chunking, and retrieval-augmented generation
- **Qdrant Vector Database**: High-performance vector search with persistent storage
- **Conversation Management**: Persistent conversations with custom system prompts
- **Extensible Architecture**: Easy to add new LLM providers, tools, and document loaders
- **Docker Ready**: Production-ready containerization with Qdrant included
- **Structured Logging**: JSON-formatted logs for production use

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- OpenAI API key or Anthropic API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd accountant-chat-bot-be

# Start Qdrant vector database (persistent storage)
docker-compose up -d qdrant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Running the Server

```bash
# Make sure Qdrant is running
docker-compose up -d qdrant

# Development mode with auto-reload
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker (Full Stack)

```bash
# Build and run everything (Qdrant + Chatbot)
docker-compose up --build

# This starts:
# - Qdrant on port 6333 (with persistent storage)
# - Chatbot API on port 8000

# View Qdrant dashboard
open http://localhost:6333/dashboard
```

## API Endpoints

### Chat

```bash
# Simple chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is depreciation in accounting?",
    "system_prompt": "You are an expert accountant."
  }'

# Continue conversation
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you give me an example?",
    "conversation_id": "<conversation-id-from-previous-response>"
  }'
```

### Conversations

```bash
# List conversations
curl http://localhost:8000/api/v1/conversations

# Get conversation details
curl http://localhost:8000/api/v1/conversations/<conversation-id>

# Create conversation with custom system prompt
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt": "You are a tax specialist."
  }'
```

### RAG (Retrieval-Augmented Generation)

```bash
# Ingest document
curl -X POST http://localhost:8000/api/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document content here...",
    "source": "tax_guide.pdf"
  }'

# Query with RAG
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the tax deductions for home office?"
  }'
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## Project Structure

```
accountant-chat-bot-be/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   ├── models.py        # Pydantic request/response models
│   │   └── dependencies.py  # Dependency injection
│   ├── core/
│   │   ├── config.py        # Configuration management
│   │   ├── logging.py       # Structured logging
│   │   └── exceptions.py    # Custom exceptions
│   ├── llm/
│   │   ├── base.py          # Base LLM interface
│   │   ├── openai_llm.py    # OpenAI implementation
│   │   ├── anthropic_llm.py # Anthropic implementation
│   │   └── factory.py       # LLM factory
│   ├── agent/
│   │   ├── agent.py         # Agentic core with tool calling
│   │   ├── tools.py         # Tool system
│   │   └── conversation.py  # Conversation management
│   ├── rag/
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── retriever.py     # Document retrieval
│   │   └── pipeline.py      # Complete RAG pipeline
│   ├── vectordb/
│   │   ├── base.py          # Vector DB interface
│   │   ├── chroma.py        # ChromaDB implementation
│   │   └── factory.py       # Vector DB factory
│   └── documents/
│       ├── base.py          # Document loader interface
│       ├── loaders.py       # File type loaders
│       └── text_splitter.py # Text chunking
├── examples/
│   ├── basic_chat.py        # Basic chat example
│   ├── chat_with_tools.py   # Tool usage example
│   └── rag_example.py       # RAG pipeline example
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Custom System Prompts

You can customize the AI's behavior by providing a system prompt:

```python
# Via API
{
  "message": "Help me with tax planning",
  "system_prompt": "You are a certified tax accountant specializing in small business taxation."
}

# Via environment variable
DEFAULT_SYSTEM_PROMPT="You are a helpful AI assistant specialized in accounting."
```

## Adding Custom Tools

```python
from app.agent.tools import BaseTool, ToolResult, tool_registry

class MyCustomTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description of what this tool does"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"]
        }

    async def execute(self, param1: str) -> ToolResult:
        # Your tool logic here
        return ToolResult(success=True, output="Result")

# Register the tool
tool_registry.register(MyCustomTool())
```

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai, anthropic) | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | gpt-4-turbo-preview |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ANTHROPIC_MODEL` | Anthropic model name | claude-3-opus-20240229 |
| `QDRANT_HOST` | Qdrant server host | localhost |
| `QDRANT_PORT` | Qdrant server port | 6333 |
| `QDRANT_COLLECTION_NAME` | Qdrant collection name | documents |
| `DEFAULT_SYSTEM_PROMPT` | Default system prompt | You are a helpful AI assistant |
| `RAG_CHUNK_SIZE` | Document chunk size | 1000 |
| `RAG_TOP_K` | Number of documents to retrieve | 5 |

See `.env.example` for all configuration options.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black app/ tests/

# Lint
ruff check app/ tests/

# Type checking
mypy app/
```

## License

MIT License