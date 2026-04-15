# Xagent Web API

## Overview

The Xagent Web API provides a FastAPI-based backend service for managing and executing AI agents. It offers RESTful endpoints and WebSocket connections for real-time agent execution monitoring and interaction.

## Features

### Agent Management
- **Agent Execution**: Create and manage agent execution tasks
- **Execution History**: Track and retrieve agent execution history
- **Nested Agents**: Support for hierarchical agent execution with parent-child relationships

### Real-time Communication
- **WebSocket Support**: Real-time agent execution monitoring and streaming
- **Event Streaming**: Live updates on agent status, tool calls, and results
- **Interactive Control**: Send commands and feedback during agent execution

### File Management
- **File Upload**: Upload files for agent processing
- **File Storage**: Organized file storage with workspace isolation
- **File Operations**: Read, write, and manage files through the API

### Visualization
- **DAG Visualization**: Generate and visualize agent execution graphs
- **Execution Flow**: Track agent execution flow and dependencies

### Observability
- **Langfuse Integration**: Built-in tracing and monitoring
- **Execution Metrics**: Track performance and resource usage
- **Error Tracking**: Comprehensive error logging and reporting

## API Endpoints

### Agent Operations
- `POST /api/agent/create` - Create a new agent execution
- `GET /api/agent/{agent_id}` - Get agent execution details
- `GET /api/agent/{agent_id}/history` - Get execution history
- `DELETE /api/agent/{agent_id}` - Delete agent execution

### WebSocket
- `WS /ws/agent/{agent_id}` - Real-time agent execution updates

### File Management
- `POST /api/files/upload` - Upload a file
- `GET /api/files/{file_id}` - Download a file
- `GET /api/files` - List uploaded files
- `DELETE /api/files/{file_id}` - Delete a file

### DAG Visualization
- `GET /api/dag/{agent_id}` - Get DAG representation of agent execution
- `POST /api/dag/validate` - Validate DAG structure

## Installation

1. Install the required dependencies:
```bash
pip install fastapi uvicorn websockets python-multipart
```

2. Ensure environment variables are set (see main project README)

## Running the Server

Start the web API server:

```bash
# Using Python module
python -m xagent.web.__main__

# Or using the start script
python start_web.py
```

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, access the interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## WebSocket Message Format

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agent/{agent_id}');
```

### Message Types
- `status`: Agent status updates (running, completed, failed)
- `tool_call`: Tool execution events
- `result`: Tool execution results
- `error`: Error messages
- `log`: Execution logs

### Example Message
```json
{
  "type": "tool_call",
  "data": {
    "tool": "calculator",
    "input": "2 + 2",
    "result": "4"
  }
}
```

## Development

### Project Structure
```
src/xagent/web/
├── __init__.py           # Package initialization
├── __main__.py           # Server entry point
├── app.py                # FastAPI application setup
├── routes/               # API route handlers
│   ├── agent.py          # Agent endpoints
│   ├── files.py          # File management endpoints
│   └── dag.py            # DAG visualization endpoints
└── websocket/            # WebSocket handlers
    └── manager.py        # WebSocket connection manager
```

### Adding New Endpoints

1. Create a new route file in `routes/`
2. Define your FastAPI router
3. Register it in `app.py`

Example:
```python
# routes/custom.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/custom", tags=["custom"])

@router.get("/endpoint")
async def custom_endpoint():
    return {"message": "Hello"}
```

## Configuration

Configuration is managed through environment variables:

- `WEB_HOST`: Server host (default: `0.0.0.0`)
- `WEB_PORT`: Server port (default: `8000`)
- `CORS_ORIGINS`: Allowed CORS origins
- `XAGENT_MAX_UPLOAD_SIZE`: Maximum per-file upload size (supports bytes or values like `100M`); nginx defers file-size enforcement to the backend so multipart overhead does not trigger premature 413 responses

## Testing

Run the web API tests:
```bash
pytest tests/web/
```

## Production Deployment

For production deployment:

1. Use a production ASGI server like Gunicorn:
```bash
gunicorn xagent.web.__main__:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. Set up proper CORS origins
3. Enable HTTPS/TLS
4. Configure proper logging and monitoring
5. Set up authentication and authorization
