# Summit MCP Server - Combined LLM Version

A Model Context Protocol (MCP) server for the Summit digital repository with support for both OpenAI and Ollama LLM backends.

## Features

### Core Functionality
- **Complete Summit Repository Access**: Fetch, download, and analyze academic content
- **Metadata Processing**: Full field extraction and categorization 
- **File Management**: Download PDFs and associated files with size limits
- **Data Quality Validation**: Comprehensive validation and error reporting
- **Search Capabilities**: Advanced search across all stored field data

### LLM Integration
- **Dual Backend Support**: Choose between OpenAI and Ollama
- **Automatic Fallback**: If primary backend fails, automatically tries the other
- **Runtime Switching**: Change LLM backends without restarting the server
- **Flexible Configuration**: Easy setup for either or both backends

### Web Interface
- **Modern UI**: Beautiful, responsive web interface
- **Real-time Status**: Live LLM backend status indicators
- **Interactive Tools**: Easy tool execution with smart argument templates
- **Result Formatting**: Human-readable output with syntax highlighting

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install fastapi uvicorn aiohttp aiofiles requests mcp fastmcp

# Optional: Install OpenAI support
pip install openai

# Optional: Install/setup Ollama
# Visit https://ollama.ai for installation instructions
```

### 2. Configuration

Edit `config.json`:

```json
{
  "llm_backend": "auto",
  "openai": {
    "api_key": "sk-your-openai-api-key-here",
    "model": "gpt-4o-mini", 
    "enabled": true
  },
  "ollama": {
    "host": "http://localhost:11434",
    "model": "llama3.2:latest",
    "enabled": true
  }
}
```

### 3. Start the Server

```bash
# Start MCP server (port 9000)
python mcp_server_combined.py

# Start web UI (port 8001) 
python web_ui_combined.py
```

### 4. Access the Interface

Open your browser to: `http://localhost:8001`

## Configuration Options

### LLM Backend Settings

| Setting | Options | Description |
|---------|---------|-------------|
| `llm_backend` | `auto`, `openai`, `ollama` | Preferred backend (auto tries OpenAI first) |
| `openai.enabled` | `true`, `false` | Enable OpenAI integration |
| `ollama.enabled` | `true`, `false` | Enable Ollama integration |

### OpenAI Configuration

```json
{
  "openai": {
    "api_key": "sk-your-key-here",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 60,
    "max_retries": 3,
    "retry_delay": 1.0
  }
}
```

### Ollama Configuration

```json
{
  "ollama": {
    "host": "http://localhost:11434",
    "model": "llama3.2:latest",
    "timeout": 120
  }
}
```

## Available Tools

### Core Tools
- `fetch_summit_nodes` - Get repository nodes with pagination
- `download_complete_node_data` - Download metadata and files
- `extract_all_field_combinations` - Extract all metadata fields
- `search_across_all_fields` - Search stored data
- `download_file` - Download specific PDFs and metadata

### Analysis Tools  
- `analyze_with_llm_batch` - Batch LLM analysis of multiple nodes
- `validate_and_report_data_quality` - Data quality validation

### Management Tools
- `get_comprehensive_server_stats` - Server statistics and LLM status
- `switch_llm_backend` - Runtime backend switching

## Command Line Usage

### Server Options

```bash
python mcp_server_combined.py --help

Options:
  --host HOST          Server host (default: 0.0.0.0)
  --port PORT          Server port (default: 9000)  
  --backend BACKEND    Force LLM backend (openai, ollama, auto)
  --disable-llm        Disable all LLM features
  --config CONFIG      Path to config file (default: config.json)
```

### Examples

```bash
# Force OpenAI backend
python mcp_server_combined.py --backend openai

# Disable LLM features entirely
python mcp_server_combined.py --disable-llm

# Use custom config file
python mcp_server_combined.py --config my_config.json
```

## Usage Examples

### Basic Node Analysis

```python
# Fetch recent nodes
result = await fetch_summit_nodes(
    updated_after="2024-01-01",
    max_nodes=50
)

# Download and analyze a specific node
result = await download_complete_node_data(
    node_id="17",
    include_files=True,
    perform_analysis=True
)
```

### Batch LLM Analysis

```python
# Analyze multiple nodes with OpenAI
result = await analyze_with_llm_batch(
    node_ids=["17", "1234", "5678"],
    analysis_types=["summary", "full"],
    backend="openai"
)

# Analyze with automatic backend selection
result = await analyze_with_llm_batch(
    node_ids=["17", "1234"],
    analysis_types=["summary"]
)
```

### Backend Management

```python
# Check LLM status
stats = await get_comprehensive_server_stats()
print(stats["llm_status"])

# Switch to Ollama
result = await switch_llm_backend(backend="ollama")
```

## Web Interface Features

### LLM Status Dashboard
- **Visual Indicators**: Color-coded badges show backend availability
- **Active Backend**: Highlighted badge shows current backend
- **Model Information**: Hover for model details
- **Runtime Switching**: Change backends without reconnecting

### Smart Tool Execution
- **Contextual Examples**: Parameter examples based on tool types
- **Real-time Validation**: JSON syntax checking
- **Result Formatting**: Syntax-highlighted, structured output
- **Error Handling**: Clear error messages and suggestions

### Activity Monitoring
- **Real-time Logs**: Live activity feed with timestamps
- **Status Tracking**: Connection and execution status
- **Performance Metrics**: Tool execution timing

## Troubleshooting

### OpenAI Issues

```bash
# Check API key
export OPENAI_API_KEY="your-key"
python -c "from openai import OpenAI; print('OK' if OpenAI().api_key else 'Missing key')"

# Test connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

### Ollama Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull required model
ollama pull llama3.2:latest

# Check logs
ollama logs
```

### Common Solutions

| Issue | Solution |
|-------|----------|
| "No LLM backend available" | Enable at least one backend in config |
| "OpenAI API key required" | Set valid API key in config.json |
| "Connection refused" (Ollama) | Start Ollama service |
| "Model not found" | Pull the model with `ollama pull` |

## Architecture

### Server Components
- **FastMCP Server**: HTTP-based MCP protocol implementation
- **Dual LLM Analyzers**: OpenAI and Ollama integrations
- **Combined Analyzer**: Automatic backend selection and fallback
- **Metadata Database**: SQLite storage for processed data
- **Summit API Client**: Async HTTP client for repository access

### Web Client
- **FastAPI Backend**: WebSocket bridge to MCP server
- **Modern Frontend**: Vanilla JavaScript with responsive design
- **Real-time Communication**: WebSocket for live updates
- **Smart UI**: Context-aware forms and result formatting

## Development

### Adding New Tools

```python
@mcp.tool()
async def my_new_tool(param: str) -> Dict[str, Any]:
    """Tool description for UI.
    
    Args:
        param: Parameter description for better examples
    """
    # Tool implementation
    return {"result": "success"}
```

### Extending LLM Support

```python
class MyLLMAnalyzer:
    async def analyze_metadata(self, metadata: Dict, analysis_type: str) -> Dict:
        return {
            "analysis": "analysis text",
            "type": analysis_type,
            "backend": "my_backend"
        }

# Register with combined analyzer
llm_analyzer.my_analyzer = MyLLMAnalyzer()
```

## License

[Include your license information here]

## Contributing

[Include contribution guidelines here]

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs for error details
3. Test individual components separately
4. File an issue with complete error information
