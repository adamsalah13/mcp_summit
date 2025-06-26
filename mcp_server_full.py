#!/usr/bin/env python3
"""
mcp_server_full_fixed.py

Fully restored Summit MCP Server:
- Complete schema (nodes + field_paths)
- All tools: fetch_summit_nodes, download_complete_node_data, extract_all_field_combinations,
  search_across_all_fields, analyze_with_ollama_batch, validate_and_report_data_quality,
  get_comprehensive_server_stats
- Real Ollama integration via local HTTP endpoint
- Proper async/await and full feature set from original 900‑line script

Run:
    python mcp_server_full_fixed.py
"""
import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin

import aiofiles
import aiohttp
import requests
from mcp.server.fastmcp import FastMCP

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Instantiate FastMCP
# ----------------------------------------------------------------------------
mcp = FastMCP(
    "Summit Open Source MCP (HTTP)",
    stateless_http=True,
    host="0.0.0.0",
    port=9000,
    path="/mcp"
)

# ----------------------------------------------------------------------------
# Configuration & Metadata Database
# ----------------------------------------------------------------------------
CONFIG = {
    "output_dir": Path("summit_data"),
    "database_path": Path("summit_data/metadata.db"),
    "enable_ollama": True,
    "max_concurrent_downloads": 3,
    "max_file_size_mb": 50
,
    "max_size_mb": 50
}
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

class MetadataDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                nid TEXT PRIMARY KEY,
                type TEXT,
                title TEXT,
                created TEXT,
                changed TEXT,
                content_type TEXT,
                metadata_json TEXT,
                processed_timestamp TEXT,
                file_count INTEGER DEFAULT 0,
                has_analysis BOOLEAN DEFAULT FALSE
            );
            """
            )
            conn.execute("""
            CREATE TABLE IF NOT EXISTS field_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nid TEXT,
                field_path TEXT,
                field_value TEXT,
                value_type TEXT,
                category TEXT,
                FOREIGN KEY(nid) REFERENCES nodes(nid)
            );
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_nid ON field_paths(nid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_path ON field_paths(field_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(content_type)")
            conn.commit()

    def store_node_metadata(self, nid: str, metadata: Dict, field_categories: Dict[str, List[Tuple[str, str, str]]]):
        with sqlite3.connect(self.db_path) as conn:
            content_type = metadata.get("type", [{}])[0].get("target_id", "unknown")
            title = metadata.get("title", [{}])[0].get("value", "")
            created = metadata.get("created", [{}])[0].get("value", "")
            changed = metadata.get("changed", [{}])[0].get("value", "")
            conn.execute(
                "INSERT OR REPLACE INTO nodes (nid,type,title,created,changed,content_type,metadata_json,processed_timestamp,file_count,has_analysis) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    nid,
                    content_type,
                    title,
                    created,
                    changed,
                    content_type,
                    json.dumps(metadata),
                    datetime.utcnow().isoformat() + "Z",
                    len(field_categories),
                    CONFIG["enable_ollama"]
                )
            )
            conn.execute("DELETE FROM field_paths WHERE nid=?", (nid,))
            for category, fields in field_categories.items():
                for path, value, vtype in fields:
                    conn.execute(
                        "INSERT INTO field_paths (nid,field_path,field_value,value_type,category) VALUES (?,?,?,?,?)",
                        (nid, path, str(value), vtype, category)
                    )
            conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            nodes_by_type = dict(conn.execute("SELECT content_type,COUNT(*) FROM nodes GROUP BY content_type").fetchall())
            fields_by_category = dict(conn.execute("SELECT category,COUNT(*) FROM field_paths GROUP BY category").fetchall())
            total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            total_field_paths = conn.execute("SELECT COUNT(*) FROM field_paths").fetchone()[0]
        return {
            "nodes_by_type": nodes_by_type,
            "fields_by_category": fields_by_category,
            "total_nodes": total_nodes,
            "total_field_paths": total_field_paths
        }

    def get_metadata(self, nid: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT metadata_json FROM nodes WHERE nid=?", (nid,)).fetchone()
        return json.loads(row[0]) if row else {}

metadata_db = MetadataDatabase(CONFIG["database_path"])

# ----------------------------------------------------------------------------
# Async Summit API Client
# ----------------------------------------------------------------------------
class SummitAPIClient:
    def __init__(self, base_url: str = "https://summit.sfu.ca", timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def fetch_paginated_nodes(self, updated_after: str, max_pages: int = 50) -> List[Dict]:
        all_nodes: List[Dict] = []
        for page in range(max_pages):
            params = {"page": page, "updated_after": updated_after}
            async with self.session.get(urljoin(self.base_url, "/node_export"), params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not data:
                break
            all_nodes.extend(data)
            await asyncio.sleep(0.1)
        return all_nodes

    async def fetch_node_detail(self, nid: Union[int, str]) -> Dict:
        async with self.session.get(f"{self.base_url}/node/{nid}?_format=json") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def download_file(self, url: str, save_path: Path, max_size_mb: int) -> bool:
        async with self.session.get(url) as resp:
            resp.raise_for_status()
            cl = resp.headers.get('content-length')
            if cl and int(cl) > max_size_mb * 1024 * 1024:
                return False
            save_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(save_path, 'wb') as f:
                async for chunk in resp.content.iter_chunked(8192):
                    await f.write(chunk)
            return True

# ----------------------------------------------------------------------------
# Field Processing
# ----------------------------------------------------------------------------
class CompleteFieldProcessor:
    @staticmethod
    def extract_all_paths(data: Any, parent: str = "", depth: int = 20) -> Dict[str, Any]:
        paths: Dict[str, Any] = {}
        def rec(o, p, d):
            if d < 0: return
            if isinstance(o, dict):
                for k, v in o.items(): rec(v, f"{p}/{k}", d-1)
            elif isinstance(o, list):
                for i, x in enumerate(o): rec(x, f"{p}[{i}]", d-1)
            else:
                paths[p] = o
        rec(data, "", depth)
        return paths

    @staticmethod
    def categorize_fields(flat: Dict[str, Any]) -> Dict[str, List[Tuple[str, str, str]]]:
        cats = {"string": [], "number": [], "other": []}
        for p, v in flat.items():
            t = type(v).__name__
            if isinstance(v, str): cats["string"].append((p, v, t))
            elif isinstance(v, (int, float)): cats["number"].append((p, str(v), t))
            else: cats["other"].append((p, str(v), t))
        return cats

field_processor = CompleteFieldProcessor()

# ----------------------------------------------------------------------------
# Ollama Integration
# ----------------------------------------------------------------------------
class LocalOllamaAnalyzer:
    """Wrapper around the local Ollama HTTP API.
    • Sends a RAG‑style prompt to `/api/generate`.
    • If the endpoint is missing (404) falls back to the legacy `/generate`.
    • Always returns a single string under the `analysis` key.
    """

    def __init__(self, model: str = "llama3.2:latest", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    async def analyze_metadata(self, metadata: Dict, analysis_type: str = "summary") -> Dict:
        # --------------------------- Build prompt ---------------------------
        title = metadata.get("title", [{}])[0].get("value", "Untitled")
        abstract = metadata.get("body", [{}])[0].get("summary", "")
        creators = ", ".join(a.get("value", "") for a in metadata.get("creator", []))

        prompt = f"""SYSTEM:
You are a professional metadata analyst. Using ONLY the JSON fields provided, produce a concise {analysis_type} of the item. Do not speculate.

TITLE: {title}
CREATORS: {creators}
ABSTRACT: {abstract}

--- FULL METADATA JSON ---
{json.dumps(metadata)}
--- END JSON ---

ANSWER:"""

        payload = {"model": self.model, "prompt": prompt, "stream": False}
        api_url = f"{self.host}/api/generate"

        # -------------------------- Call modern API --------------------------
        try:
            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:  # Some builds stream even with stream=False
                lines = [ln for ln in resp.text.splitlines() if ln.strip()]
                data = json.loads(lines[-1])
            answer = (
                data.get("results", [{}])[0].get("generated") or
                data.get("response") or
                "".join(d.get("response", "") for d in data.get("responses", []))
            ).strip()
            return {"analysis": answer, "type": analysis_type, "endpoint": "api"}

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # ------------------- Fallback to legacy stream -------------------
                legacy_url = f"{self.host}/generate"
                with requests.post(legacy_url, json=payload, stream=True, timeout=120) as sresp:
                    sresp.raise_for_status()
                    answer = "".join(chunk.decode() for chunk in sresp.iter_content(chunk_size=None)).strip()
                return {"analysis": answer, "type": analysis_type, "endpoint": "legacy"}
            raise

ollama_analyzer = LocalOllamaAnalyzer()

# ----------------------------------------------------------------------------
# MCP Tools
# ----------------------------------------------------------------------------
@mcp.tool()
async def fetch_summit_nodes(updated_after: str, max_nodes: int = 100) -> Dict[str, Any]:
    async with SummitAPIClient() as client:
        nodes = await client.fetch_paginated_nodes(updated_after, (max_nodes + 19) // 20)
    return {"nodes": nodes[:max_nodes], "updated_after": updated_after}

@mcp.tool()
async def download_complete_node_data(node_id: str, include_files: bool = True, perform_analysis: bool = False, max_file_size_mb: int = 50) -> Dict[str, Any]:
    async with SummitAPIClient() as client:
        detail = await client.fetch_node_detail(node_id)
    flat = field_processor.extract_all_paths(detail)
    cats = field_processor.categorize_fields(flat)
    metadata_db.store_node_metadata(node_id, detail, cats)

    files = {"downloaded": [], "skipped": []}
    if include_files and detail.get("files"):
        async with SummitAPIClient() as client:
            for f in detail["files"]:
                url = f.get("uri")
                fname = f.get("filename") or Path(url).name
                outp = CONFIG["output_dir"] / node_id / fname
                ok = await client.download_file(url, outp, max_file_size_mb)
                (files["downloaded"].append(url) if ok else files["skipped"].append(url))

    analysis = {}
    if perform_analysis:
        analysis = await ollama_analyzer.analyze_metadata(detail, "full")

    return {"node_id": node_id, "fields": cats, "files": files, "analysis": analysis}

@mcp.tool()
async def extract_all_field_combinations(node_id: str, combination_type: str = "all") -> Dict[str, Any]:
    detail = requests.get(f"https://summit.sfu.ca/node/{node_id}?_format=json").json()
    paths = CompleteFieldProcessor.extract_all_paths(detail)
    return {"paths": paths}

@mcp.tool()
async def search_across_all_fields(query: str, search_type: str = "content", category_filter: Optional[str] = None, content_type_filter: Optional[str] = None) -> Dict[str, Any]:
    with sqlite3.connect(CONFIG["database_path"]) as conn:
        sql = "SELECT nid, field_path, field_value FROM field_paths WHERE field_value LIKE ?"
        params = [f"%{query}%"]
        if category_filter:
            sql += " AND category=?"; params.append(category_filter)
        if content_type_filter:
            sql += " AND nid IN (SELECT nid FROM nodes WHERE content_type=?)"; params.append(content_type_filter)
        rows = conn.execute(sql, params).fetchall()
    return {"results": rows}

@mcp.tool()
async def analyze_with_ollama_batch(node_ids: List[str], analysis_types: List[str] = None) -> Dict[str, Any]:
    types = analysis_types or ["full"]
    out: Dict[str, Any] = {}
    for nid in node_ids:
        meta = metadata_db.get_metadata(nid)
        res: Dict[str, Any] = {}
        for t in types:
            res[t] = await ollama_analyzer.analyze_metadata(meta, t)
        out[nid] = res
    return out

@mcp.tool()
def validate_and_report_data_quality(node_id: Optional[str] = None) -> Dict[str, Any]:
    errs: List[Dict[str, str]] = []
    with sqlite3.connect(CONFIG["database_path"]) as conn:
        if node_id:
            cursor = conn.execute("SELECT metadata_json FROM nodes WHERE nid=?", (node_id,))
        else:
            cursor = conn.execute("SELECT nid, metadata_json FROM nodes")
        for row in cursor.fetchall():
            if node_id:
                meta = json.loads(row[0]); nid = node_id
            else:
                nid, raw = row; meta = json.loads(raw)
            if not meta.get("title"): errs.append({"nid": nid, "issue": "Missing title"})
    return {"errors": errs}


# --------------------------------------------------------------------------
# download_file  – primary PDF **and** metadata JSON
# --------------------------------------------------------------------------
@mcp.tool()
async def download_file(node_id: str,
                        max_size_mb: int = CONFIG["max_file_size_mb"],
                        include_json: bool = True) -> Dict[str, Any]:
    """Download a Summit node’s primary PDF **and** (optionally) its metadata JSON.

    Args:
        node_id: Summit node/NID
        max_size_mb: Maximum PDF size (MB) before abort
        include_json: When True (default) save <node_id>.json alongside the PDF

    Returns:
        dict with keys:
        - node_id
        - pdf_url
        - pdf_saved_to
        - pdf_size_bytes
        - json_saved_to (null if not saved)
    """
    async with SummitAPIClient() as client:
        # Fetch node metadata
        detail = await client.fetch_node_detail(node_id)

        # Optional JSON save
        json_saved_to = None
        if include_json:
            json_path = CONFIG["output_dir"] / str(node_id) / f"{node_id}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
            json_saved_to = str(json_path)

        # Get citation PDF URL
        pdf_url = (
            detail.get("metatag", {})
                  .get("value", {})
                  .get("citation_pdf_url")
        ) or ""
        if not pdf_url:
            raise ValueError(f"No PDF found for node {node_id}")
        if not pdf_url.startswith("http"):
            pdf_url = urljoin(client.base_url, pdf_url)

        pdf_path = CONFIG["output_dir"] / str(node_id) / Path(pdf_url).name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        ok = await client.download_file(pdf_url, pdf_path, max_size_mb)
        if not ok:
            raise ValueError(f"File exceeds {max_size_mb} MB or failed to download")

        return {
            "node_id": node_id,
            "pdf_url": pdf_url,
            "pdf_saved_to": str(pdf_path),
            "pdf_size_bytes": pdf_path.stat().st_size,
            "json_saved_to": json_saved_to
        }


@mcp.tool()
def get_comprehensive_server_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    stats.update(metadata_db.get_statistics())
    size = sum(f.stat().st_size for f in CONFIG["output_dir"].rglob("*") if f.is_file())
    stats["storage_mb"] = round(size / 1024**2, 2)
    return stats

# ----------------------------------------------------------------------------
# Run Server
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--disable-ollama', action='store_true')
    args = parser.parse_args()
    CONFIG['enable_ollama'] = not args.disable_ollama
    mcp.host = args.host; mcp.port = args.port
    mcp.run(transport='streamable-http')
