#!/usr/bin/env python3
"""
web_ui_combined.py

Complete Summit MCP Web Client UI with Combined LLM Support
• Connects to your Streamable HTTP FastMCP server at /mcp
• WebSocket bridge under /ws
• Connect/Disconnect, dynamic tool list with descriptions, JSON argument templates, 
  results pane, and activity log
• Enhanced UI to show LLM integration status (OpenAI/Ollama)
• Backend switching capabilities

Run:
python web_ui_combined.py
"""

import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# ----------------------------------------------------------------------------
# Configure HTTP-based MCP client
# ----------------------------------------------------------------------------

MCP_URL = "http://127.0.0.1:9000/mcp"
transport = StreamableHttpTransport(url=MCP_URL)
http_client = Client(transport)

# ----------------------------------------------------------------------------
# FastAPI app with lifespan: init/teardown of HTTP JSON-RPC client
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await http_client.__aenter__()
    yield
    await http_client.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("web_ui_combined")

# ----------------------------------------------------------------------------
# HTML UI endpoint
# ----------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Summit MCP Server - Combined LLM</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .header h1 {
            color: #333;
            margin: 0;
            font-size: 2.5em;
        }
        .header .subtitle {
            color: #666;
            font-size: 1.1em;
            margin-top: 10px;
        }
        .llm-badges {
            margin-top: 15px;
        }
        .llm-badge {
            display: inline-block;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin: 0 5px;
        }
        .llm-badge.openai { background: #10a37f; }
        .llm-badge.ollama { background: #2563eb; }
        .llm-badge.inactive { background: #6b7280; }
        .llm-badge.active { box-shadow: 0 0 10px rgba(0,0,0,0.3); }
        .status-section {
            margin-bottom: 25px;
        }
        .status {
            font-weight: bold;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .btn-primary {
            background: #667eea;
            color: white;
        }
        .btn-primary:hover {
            background: #5a6fd8;
            transform: translateY(-1px);
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .btn-success {
            background: #10a37f;
            color: white;
        }
        .btn-success:hover {
            background: #0d8f6f;
        }
        .btn-info {
            background: #2563eb;
            color: white;
        }
        .btn-info:hover {
            background: #1d4ed8;
        }
        .btn-warning {
            background: #f59e0b;
            color: white;
        }
        .btn-warning:hover {
            background: #d97706;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }
        .panel {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }
        .panel h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .tool-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .tool-item {
            background: white;
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tool-item:hover {
            background: #f0f0f0;
            transform: translateX(5px);
        }
        .tool-name {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .tool-description {
            color: #666;
            font-size: 0.9em;
        }
        .execution-area {
            background: white;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }
        textarea, input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            resize: vertical;
        }
        .results {
            background: #1e1e1e;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            margin-top: 15px;
        }
        .activity-log {
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 6px;
        }
        .log-entry {
            margin-bottom: 8px;
            padding: 5px;
            border-left: 3px solid #667eea;
            padding-left: 10px;
        }
        .log-success { border-left-color: #10a37f; }
        .log-error { border-left-color: #e53e3e; }
        .log-info { border-left-color: #3182ce; }
        .llm-controls {
            background: #f1f5f9;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .backend-selector {
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }
        .backend-status {
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Summit MCP Server</h1>
            <div class="subtitle">
                Model Context Protocol Server for Summit Repository
            </div>
            <div class="llm-badges">
                <span id="openai-badge" class="llm-badge openai inactive">OpenAI</span>
                <span id="ollama-badge" class="llm-badge ollama inactive">Ollama</span>
            </div>
        </div>

        <div class="status-section">
            <div id="status" class="status disconnected">
                Status: <strong>Disconnected</strong>
            </div>
            <div class="controls">
                <button id="connectBtn" class="btn-primary">Connect to MCP Server</button>
                <button id="disconnectBtn" class="btn-secondary" disabled>Disconnect</button>
                <button id="refreshToolsBtn" class="btn-success" disabled>Refresh Tools</button>
                <button id="getStatsBtn" class="btn-success" disabled>Get Server Stats</button>
            </div>
        </div>

        <div class="llm-controls" id="llmControls" style="display: none;">
            <h4 style="margin-top: 0;">LLM Backend Control</h4>
            <div class="backend-selector">
                <label for="backendSelect">Active Backend:</label>
                <select id="backendSelect">
                    <option value="auto">Auto</option>
                    <option value="openai">OpenAI</option>
                    <option value="ollama">Ollama</option>
                </select>
                <button id="switchBackendBtn" class="btn-info">Switch Backend</button>
            </div>
            <div id="backendStatus" class="backend-status">Backend status will appear here</div>
        </div>

        <div class="grid">
            <div class="panel">
                <h3>Available Tools</h3>
                <div id="toolList" class="tool-list">
                    <div style="color: #666; text-align: center; padding: 20px;">
                        Connect to server to load tools
                    </div>
                </div>
            </div>

            <div class="panel">
                <h3>Tool Execution</h3>
                <div class="execution-area">
                    <div class="form-group">
                        <label for="selectedTool">Selected Tool:</label>
                        <input type="text" id="selectedTool" readonly placeholder="Select a tool from the list">
                    </div>
                    <div class="form-group">
                        <label for="toolArgs">Arguments (JSON):</label>
                        <textarea id="toolArgs" rows="4" placeholder='{"example": "value"}'></textarea>
                    </div>
                    <button id="executeBtn" class="btn-primary" disabled>Execute Tool</button>
                </div>
            </div>
        </div>

        <div class="panel">
            <h3>Results</h3>
            <div id="results" class="results">Ready to execute tools...</div>
        </div>

        <div class="panel">
            <h3>Activity Log</h3>
            <div id="activityLog" class="activity-log"></div>
        </div>
    </div>

    <script>
        class MCPClient {
            constructor() {
                this.ws = null;
                this.connected = false;
                this.tools = [];
                this.selectedTool = null;
                this.llmStatus = null;
                this.setupEventListeners();
                this.addLog('Application started', 'info');
            }

            setupEventListeners() {
                document.getElementById('connectBtn').addEventListener('click', () => this.connect());
                document.getElementById('disconnectBtn').addEventListener('click', () => this.disconnect());
                document.getElementById('refreshToolsBtn').addEventListener('click', () => this.loadTools());
                document.getElementById('getStatsBtn').addEventListener('click', () => this.getServerStats());
                document.getElementById('executeBtn').addEventListener('click', () => this.executeTool());
                document.getElementById('switchBackendBtn').addEventListener('click', () => this.switchBackend());
            }

            connect() {
                this.addLog('Attempting to connect to MCP server...', 'info');
                this.ws = new WebSocket('ws://localhost:8001/ws');
                
                this.ws.onopen = () => {
                    this.connected = true;
                    this.updateStatus(true);
                    this.addLog('Connected to MCP server successfully', 'success');
                    this.loadTools();
                    this.getServerStats(); // Load LLM status
                };

                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };

                this.ws.onclose = () => {
                    this.connected = false;
                    this.updateStatus(false);
                    this.addLog('Disconnected from MCP server', 'error');
                };

                this.ws.onerror = (error) => {
                    this.addLog(`WebSocket error: ${error}`, 'error');
                };
            }

            disconnect() {
                if (this.ws) {
                    this.ws.close();
                }
            }

            updateStatus(connected) {
                const status = document.getElementById('status');
                const connectBtn = document.getElementById('connectBtn');
                const disconnectBtn = document.getElementById('disconnectBtn');
                const refreshBtn = document.getElementById('refreshToolsBtn');
                const statsBtn = document.getElementById('getStatsBtn');
                const executeBtn = document.getElementById('executeBtn');
                const llmControls = document.getElementById('llmControls');

                if (connected) {
                    status.className = 'status connected';
                    status.innerHTML = 'Status: <strong>Connected</strong>';
                    connectBtn.disabled = true;
                    disconnectBtn.disabled = false;
                    refreshBtn.disabled = false;
                    statsBtn.disabled = false;
                    llmControls.style.display = 'block';
                } else {
                    status.className = 'status disconnected';
                    status.innerHTML = 'Status: <strong>Disconnected</strong>';
                    connectBtn.disabled = false;
                    disconnectBtn.disabled = true;
                    refreshBtn.disabled = true;
                    statsBtn.disabled = true;
                    executeBtn.disabled = true;
                    llmControls.style.display = 'none';
                    this.updateLLMBadges(null);
                }
            }

            updateLLMBadges(llmStatus) {
                const openaiBadge = document.getElementById('openai-badge');
                const ollamaBadge = document.getElementById('ollama-badge');
                
                if (!llmStatus) {
                    openaiBadge.className = 'llm-badge openai inactive';
                    ollamaBadge.className = 'llm-badge ollama inactive';
                    return;
                }

                // Update OpenAI badge
                if (llmStatus.openai_available) {
                    openaiBadge.className = `llm-badge openai ${llmStatus.active_backend === 'openai' ? 'active' : ''}`;
                    openaiBadge.title = `Model: ${llmStatus.openai_model || 'N/A'}`;
                } else {
                    openaiBadge.className = 'llm-badge openai inactive';
                    openaiBadge.title = 'Not available';
                }

                // Update Ollama badge
                if (llmStatus.ollama_available) {
                    ollamaBadge.className = `llm-badge ollama ${llmStatus.active_backend === 'ollama' ? 'active' : ''}`;
                    ollamaBadge.title = `Model: ${llmStatus.ollama_model || 'N/A'}`;
                } else {
                    ollamaBadge.className = 'llm-badge ollama inactive';
                    ollamaBadge.title = 'Not available';
                }

                // Update backend selector
                const backendSelect = document.getElementById('backendSelect');
                backendSelect.value = llmStatus.active_backend || 'auto';

                // Update status text
                const backendStatus = document.getElementById('backendStatus');
                const availableBackends = [];
                if (llmStatus.openai_available) availableBackends.push('OpenAI');
                if (llmStatus.ollama_available) availableBackends.push('Ollama');
                
                if (availableBackends.length === 0) {
                    backendStatus.textContent = 'No LLM backends available';
                } else {
                    backendStatus.textContent = `Available: ${availableBackends.join(', ')} | Active: ${llmStatus.active_backend || 'None'}`;
                }
            }

            async loadTools() {
                try {
                    this.addLog('Loading available tools...', 'info');
                    this.sendMessage({
                        action: 'list_tools'
                    });
                } catch (error) {
                    this.addLog(`Failed to load tools: ${error}`, 'error');
                }
            }

            async getServerStats() {
                try {
                    this.addLog('Fetching server statistics...', 'info');
                    this.sendMessage({
                        action: 'call_tool',
                        tool_name: 'get_comprehensive_server_stats',
                        arguments: {}
                    });
                } catch (error) {
                    this.addLog(`Failed to get server stats: ${error}`, 'error');
                }
            }

            async switchBackend() {
                try {
                    const backendSelect = document.getElementById('backendSelect');
                    const newBackend = backendSelect.value;
                    
                    this.addLog(`Switching to ${newBackend} backend...`, 'info');
                    this.sendMessage({
                        action: 'call_tool',
                        tool_name: 'switch_llm_backend',
                        arguments: { backend: newBackend }
                    });
                } catch (error) {
                    this.addLog(`Failed to switch backend: ${error}`, 'error');
                }
            }

            sendMessage(message) {
                if (this.ws && this.connected) {
                    this.ws.send(JSON.stringify(message));
                } else {
                    this.addLog('Not connected to server', 'error');
                }
            }

            handleMessage(data) {
                switch (data.type) {
                    case 'tools_list':
                        this.displayTools(data.tools);
                        break;
                    case 'tool_result':
                        this.displayResult(data.result);
                        // Check if this was a stats call to update LLM status
                        if (data.result && data.result.llm_status) {
                            this.llmStatus = data.result.llm_status;
                            this.updateLLMBadges(this.llmStatus);
                        }
                        // Check if this was a backend switch call
                        if (data.result && data.result.success !== undefined) {
                            if (data.result.success) {
                                this.addLog(`Backend switched to ${data.result.current_backend}`, 'success');
                                // Refresh stats to update UI
                                setTimeout(() => this.getServerStats(), 500);
                            } else {
                                this.addLog(`Backend switch failed: ${data.result.error}`, 'error');
                            }
                        }
                        break;
                    case 'error':
                        this.addLog(`Server error: ${data.message}`, 'error');
                        break;
                    default:
                        this.displayResult(data);
                }
            }

            displayTools(tools) {
                this.tools = tools;
                const toolList = document.getElementById('toolList');
                
                if (tools.length === 0) {
                    toolList.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No tools available</div>';
                    return;
                }

                toolList.innerHTML = tools.map(tool => `
                    <div class="tool-item" onclick="client.selectTool('${tool.name}')">
                        <div class="tool-name">${tool.name}</div>
                        <div class="tool-description">${tool.description || 'No description available'}</div>
                    </div>
                `).join('');

                this.addLog(`Loaded ${tools.length} tools`, 'success');
            }

            selectTool(toolName) {
                this.selectedTool = toolName;
                document.getElementById('selectedTool').value = toolName;
                document.getElementById('executeBtn').disabled = false;
                
                // Find tool and populate example arguments
                const tool = this.tools.find(t => t.name === toolName);
                if (tool && tool.inputSchema) {
                    const exampleArgs = this.generateExampleArgs(tool.inputSchema);
                    document.getElementById('toolArgs').value = JSON.stringify(exampleArgs, null, 2);
                }
                
                this.addLog(`Selected tool: ${toolName}`, 'info');
            }

            generateExampleArgs(schema) {
                const props = schema.properties || {};
                const example = {};
                
                for (const [key, prop] of Object.entries(props)) {
                    // Use more meaningful examples based on parameter names and types
                    switch (key) {
                        case 'node_id':
                            example[key] = '17';
                            break;
                        case 'node_ids':
                            example[key] = ['17', '1234'];
                            break;
                        case 'updated_after':
                            example[key] = '2024-01-01';
                            break;
                        case 'query':
                            example[key] = 'machine learning';
                            break;
                        case 'max_nodes':
                            example[key] = 50;
                            break;
                        case 'max_file_size_mb':
                        case 'max_size_mb':
                            example[key] = 25;
                            break;
                        case 'category_filter':
                            example[key] = 'string';
                            break;
                        case 'content_type_filter':
                            example[key] = 'article';
                            break;
                        case 'search_type':
                            example[key] = 'content';
                            break;
                        case 'combination_type':
                            example[key] = 'all';
                            break;
                        case 'analysis_types':
                            example[key] = ['summary', 'full'];
                            break;
                        case 'backend':
                            example[key] = 'auto';
                            break;
                        default:
                            // Fallback to type-based examples
                            switch (prop.type) {
                                case 'string':
                                    example[key] = prop.example || (key.includes('id') ? '17' : `example_${key}`);
                                    break;
                                case 'integer':
                                case 'number':
                                    example[key] = prop.example || (key.includes('max') ? 50 : 123);
                                    break;
                                case 'boolean':
                                    example[key] = prop.example !== undefined ? prop.example : true;
                                    break;
                                case 'array':
                                    if (key.includes('id')) {
                                        example[key] = ['17', '1234'];
                                    } else if (key.includes('type')) {
                                        example[key] = ['summary', 'full'];
                                    } else {
                                        example[key] = [];
                                    }
                                    break;
                                default:
                                    example[key] = null;
                            }
                    }
                }
                
                return example;
            }

            async executeTool() {
                if (!this.selectedTool) {
                    this.addLog('No tool selected', 'error');
                    return;
                }

                try {
                    const argsText = document.getElementById('toolArgs').value.trim();
                    const args = argsText ? JSON.parse(argsText) : {};
                    
                    this.addLog(`Executing tool: ${this.selectedTool}`, 'info');
                    
                    this.sendMessage({
                        action: 'call_tool',
                        tool_name: this.selectedTool,
                        arguments: args
                    });
                    
                } catch (error) {
                    this.addLog(`Failed to execute tool: ${error}`, 'error');
                }
            }

            displayResult(result) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = this.formatResult(result);
                this.addLog('Tool execution completed', 'success');
            }

            formatResult(result) {
                if (!result) return '<div style="color: #888;">No result returned</div>';
                
                // Handle array of results (common for MCP tool responses)
                if (Array.isArray(result)) {
                    if (result.length === 0) {
                        return '<div style="color: #888;">Empty result array</div>';
                    }
                    
                    return result.map((item, index) => {
                        if (typeof item === 'string') {
                            return `<div style="margin-bottom: 15px;">
                                <div style="color: #10a37f; font-weight: bold; margin-bottom: 5px;">Result ${index + 1}:</div>
                                <div style="background: #2d3748; padding: 15px; border-radius: 6px; white-space: pre-wrap; line-height: 1.4;">${this.escapeHtml(item)}</div>
                            </div>`;
                        } else {
                            return `<div style="margin-bottom: 15px;">
                                <div style="color: #10a37f; font-weight: bold; margin-bottom: 5px;">Result ${index + 1} (Object):</div>
                                <div style="background: #2d3748; padding: 15px; border-radius: 6px;">
                                    <pre style="margin: 0; color: #e2e8f0; font-size: 13px; line-height: 1.4;">${this.escapeHtml(JSON.stringify(item, null, 2))}</pre>
                                </div>
                            </div>`;
                        }
                    }).join('');
                }
                
                // Handle single string result
                if (typeof result === 'string') {
                    return `<div style="background: #2d3748; padding: 15px; border-radius: 6px; white-space: pre-wrap; line-height: 1.4; color: #e2e8f0;">${this.escapeHtml(result)}</div>`;
                }
                
                // Handle object result
                if (typeof result === 'object') {
                    // Check if it's an error object
                    if (result.error) {
                        return `<div style="background: #742a2a; color: #fed7d7; padding: 15px; border-radius: 6px; border-left: 4px solid #e53e3e;">
                            <div style="font-weight: bold; margin-bottom: 8px;">Error:</div>
                            <div style="white-space: pre-wrap;">${this.escapeHtml(result.error)}</div>
                        </div>`;
                    }
                    
                    // Check if it's a structured response with content
                    if (result.content && Array.isArray(result.content)) {
                        return result.content.map((item, index) => {
                            if (item.type === 'text') {
                                return `<div style="margin-bottom: 15px;">
                                    <div style="color: #10a37f; font-weight: bold; margin-bottom: 5px;">Content ${index + 1}:</div>
                                    <div style="background: #2d3748; padding: 15px; border-radius: 6px; white-space: pre-wrap; line-height: 1.4; color: #e2e8f0;">${this.escapeHtml(item.text)}</div>
                                </div>`;
                            } else {
                                return `<div style="margin-bottom: 15px;">
                                    <div style="color: #10a37f; font-weight: bold; margin-bottom: 5px;">Content ${index + 1} (${item.type}):</div>
                                    <div style="background: #2d3748; padding: 15px; border-radius: 6px;">
                                        <pre style="margin: 0; color: #e2e8f0; font-size: 13px; line-height: 1.4;">${this.escapeHtml(JSON.stringify(item, null, 2))}</pre>
                                    </div>
                                </div>`;
                            }
                        }).join('');
                    }
                    
                    // Generic object formatting
                    return `<div style="background: #2d3748; padding: 15px; border-radius: 6px;">
                        <pre style="margin: 0; color: #e2e8f0; font-size: 13px; line-height: 1.4;">${this.escapeHtml(JSON.stringify(result, null, 2))}</pre>
                    </div>`;
                }
                
                // Fallback for other types
                return `<div style="background: #2d3748; padding: 15px; border-radius: 6px; color: #e2e8f0;">${this.escapeHtml(String(result))}</div>`;
            }

            escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            addLog(message, type = 'info') {
                const log = document.getElementById('activityLog');
                const timestamp = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.className = `log-entry log-${type}`;
                entry.textContent = `[${timestamp}] ${message}`;
                log.appendChild(entry);
                log.scrollTop = log.scrollHeight;
            }
        }

        // Initialize the client
        const client = new MCPClient();
    </script>
</body>
</html>
    """)

# ----------------------------------------------------------------------------
# WebSocket endpoint for real-time communication
# ----------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["action"] == "list_tools":
                try:
                    tools = await http_client.list_tools()
                    await websocket.send_text(json.dumps({
                        "type": "tools_list",
                        "tools": [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in tools]
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Failed to list tools: {str(e)}"
                    }))
                    
            elif message["action"] == "call_tool":
                try:
                    result = await http_client.call_tool(
                        message["tool_name"],
                        message.get("arguments", {})
                    )
                    await websocket.send_text(json.dumps({
                        "type": "tool_result",
                        "result": [item.text if hasattr(item, 'text') else str(item) for item in result]
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Failed to execute tool: {str(e)}"
                    }))
                    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
