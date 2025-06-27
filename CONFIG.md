# Summit MCP Server Configuration

## Setup Instructions

1. **Copy the example configuration:**
   ```bash
   cp config.json.example config.json
   ```

2. **Edit config.json and add your OpenAI API key:**
   ```json
   {
     "openai": {
       "api_key": "sk-your-actual-openai-api-key-here",
       "enabled": true
     }
   }
   ```

3. **Run the server:**
   ```bash
   python mcp_server_full_OpenAI.py
   ```

## Configuration Options

### OpenAI Settings
- `api_key`: Your OpenAI API key (required for OpenAI features)
- `model`: OpenAI model to use (default: "gpt-4o-mini")
- `temperature`: Response randomness (0.0-1.0, default: 0.7)
- `max_tokens`: Maximum response length (default: 2000)
- `enabled`: Enable/disable OpenAI features (default: true)

### Server Settings  
- `output_dir`: Directory for downloaded files (default: "summit_data")
- `max_file_size_mb`: Maximum file size for downloads (default: 50)

## Environment Variable Fallback

If no config.json exists, the server will look for the OpenAI API key in the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
python mcp_server_full_OpenAI.py
```

## Command Line Options

- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 9000)  
- `--disable-openai`: Disable OpenAI features
- `--config`: Path to config file (default: config.json)
