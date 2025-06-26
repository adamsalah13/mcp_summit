#!/usr/bin/env python3
"""
web_ui_full.py

Complete Summit MCP Web Client UI
• Connects to your Streamable HTTP FastMCP server at /mcp
• WebSocket bridge under /ws
• Connect/Disconnect, dynamic tool list with descriptions,
  JSON argument templates, results pane, and activity log

Run:
    python web_ui_full.py
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
logger = logging.getLogger("web_ui_full")

# ----------------------------------------------------------------------------
# HTML UI endpoint
# ----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Summit MCP Web UI</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    body { background: #f0f2f5; padding: 1rem; }
    .card { margin-bottom: 1rem; }
    .log { background: #fff; border: 1px solid #ddd; height: 200px;
           overflow-y: auto; font-family: monospace; padding: 0.5rem; }
    #results { background: #fff; border: 1px solid #ddd;
               padding: 0.5rem; min-height: 150px;
               font-family: monospace; white-space: pre-wrap; }
  </style>
</head>
<body>
  <nav class="navbar navbar-dark bg-primary mb-4">
    <div class="container-fluid">
      <span class="navbar-brand">🚀 Summit MCP Web UI</span>
      <span id="status" class="text-white">● Disconnected</span>
    </div>
  </nav>

  <div class="row">
    <div class="col-md-4">
      <div class="card">
        <div class="card-header">Connection</div>
        <div class="card-body">
          <button id="connect" class="btn btn-success w-100">Connect</button>
          <button id="disconnect" class="btn btn-danger w-100" disabled>Disconnect</button>
          <p id="conn-status" class="mt-2">Status: <strong>Disconnected</strong></p>
        </div>
      </div>
      <div class="card">
        <div class="card-header">Tools</div>
        <ul id="tool-list" class="list-group list-group-flush">
          <li class="list-group-item text-muted">Connect to load tools</li>
        </ul>
      </div>
    </div>
    <div class="col-md-8">
      <div class="card mb-3">
        <div class="card-header">Execute Tool</div>
        <div class="card-body">
          <select id="tool-select" class="form-select mb-2" disabled>
            <option value="">-- choose a tool --</option>
          </select>
          <textarea id="args" class="form-control mb-2" rows="4" disabled></textarea>
          <button id="run" class="btn btn-primary w-100" disabled>Run</button>
        </div>
      </div>

      <div class="card mb-3">
        <div class="card-header">Results</div>
        <div id="results" class="card-body">No results</div>
      </div>
      <div class="card">
        <div class="card-header">Activity Log</div>
        <div id="log" class="card-body log"></div>
      </div>
    </div>
  </div>

  <script>
    let ws;
    const logEl = document.getElementById('log');
    const statusEl = document.getElementById('status');
    const connStatus = document.getElementById('conn-status');
    const btnConnect = document.getElementById('connect');
    const btnDisconnect = document.getElementById('disconnect');
    const toolListEl = document.getElementById('tool-list');
    const selectTool = document.getElementById('tool-select');
    const argsEl = document.getElementById('args');
    const btnRun = document.getElementById('run');
    const resultsEl = document.getElementById('results');

    // Predefined argument templates for required tools
    const defaultTemplates = {
      fetch_summit_nodes: { updated_after: new Date().toISOString(), max_nodes: 100 },
      download_complete_node_data: { node_id: "", include_files: true, perform_analysis: false, max_file_size_mb: 50 },
      extract_all_field_combinations: { node_id: "", combination_type: "all" },
      search_across_all_fields: { query: "", search_type: "content", category_filter: null, content_type_filter: null },
      analyze_with_ollama_batch: { node_ids: [""], analysis_types: ["full"] },
      validate_and_report_data_quality: { node_id: null },
      get_comprehensive_server_stats: {},
  "download_file": { node_id: "", max_size_mb: 50, include_json: true }
};

    let defaultArgs = {};

    function appendLog(msg) {
      const t = new Date().toLocaleTimeString();
      logEl.textContent += `[${t}] ${msg}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setConnected(yes) {
      statusEl.textContent = yes ? '● Connected' : '● Disconnected';
      connStatus.innerHTML = `Status: <strong>${yes ? 'Connected' : 'Disconnected'}</strong>`;
      btnConnect.disabled = yes;
      btnDisconnect.disabled = !yes;
      selectTool.disabled = !yes;
      btnRun.disabled = !yes;
    }

    btnConnect.onclick = () => {
      ws = new WebSocket((location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + location.host + '/ws');
      ws.onopen = () => { appendLog('WebSocket opened'); ws.send(JSON.stringify({ type: 'connect' })); };
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === 'connection_status') {
          setConnected(m.connected);
          toolListEl.innerHTML = '';
          selectTool.innerHTML = '<option value="">-- choose a tool --</option>';
          defaultArgs = {};
          m.tools.forEach(t => {
            toolListEl.innerHTML += `<li class="list-group-item"><strong>${t.name}</strong><br/><small>${t.description}</small></li>`;
            selectTool.innerHTML += `<option>${t.name}</option>`;
            // Use our predefined template if available
            defaultArgs[t.name] = defaultTemplates[t.name]
              ? JSON.stringify(defaultTemplates[t.name], null, 2)
              : '{}';
          });
          appendLog('Connected to server');
        } else if (m.type === 'tool_result') {
          resultsEl.textContent = JSON.stringify(m.result, null, 2);
          appendLog('Tool result received');
        } else if (m.type === 'log') {
          appendLog(m.message);
        } else if (m.type === 'error') {
          appendLog('Error: ' + m.message);
        }
      };
      ws.onerror = () => appendLog('WebSocket error');
      ws.onclose = () => setConnected(false);
    };

    btnDisconnect.onclick = () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'disconnect' }));
      }
    };

    selectTool.onchange = () => {
      const tool = selectTool.value;
      const template = defaultTemplates[tool] || {};
      if (Object.keys(template).length === 0) {
        argsEl.value = '';
        argsEl.disabled = true;
      } else {
        argsEl.disabled = false;
        argsEl.value = JSON.stringify(template, null, 2);
      }
    };

    btnRun.onclick = () => {
      const tn = selectTool.value;
      let a;
      try { a = JSON.parse(argsEl.value || '{}'); } catch (e) { return alert('Invalid JSON: ' + e.message); }
      appendLog(`Running ${tn}...`);
      ws.send(JSON.stringify({ type: 'call_tool', tool_name: tn, arguments: a }));
    };

    // Initialize UI
    selectTool.disabled = true;
    argsEl.disabled = true;
    btnRun.disabled = true;
    setConnected(false);
    appendLog('UI ready');
  </script>
</body>
</html>
"""
)

# ----------------------------------------------------------------------------
# WebSocket bridge to HTTP JSON-RPC
# ----------------------------------------------------------------------------
@app.websocket('/ws')
async def ws_bridge(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg['type'] == 'connect':
                resp = await http_client.list_tools()
                tools = resp.tools if hasattr(resp, 'tools') else list(resp)
                await ws.send_json({'type': 'connection_status', 'connected': True, 'tools': [
                    {'name': t.name, 'description': t.description or ''} for t in tools
                ]})
            elif msg['type'] == 'disconnect':
                await ws.send_json({'type': 'connection_status', 'connected': False})
            elif msg['type'] == 'call_tool':
                tn = msg['tool_name']
                args = msg.get('arguments', {})
                await ws.send_json({'type': 'log', 'message': f'Calling {tn}...'})
                try:
                    raw = await http_client.call_tool(tn, args)
                    result = raw
                    if isinstance(raw, list) and hasattr(raw[0], 'text'):
                        try:
                            result = json.loads(raw[0].text)
                        except Exception:
                            result = raw[0].text
                    await ws.send_json({'type': 'tool_result', 'result': result})
                    await ws.send_json({'type': 'log', 'message': f'✅ {tn} completed'})
                except Exception as e:
                    await ws.send_json({'type': 'error', 'message': str(e)})
    except WebSocketDisconnect:
        pass

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
