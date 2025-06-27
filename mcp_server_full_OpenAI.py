#!/usr/bin/env python3
"""
mcp_server_full_openai.py

Summit MCP Server with OpenAI Integration:
- Complete schema (nodes + field_paths)
- All tools: fetch_summit_nodes, download_complete_node_data, extract_all_field_combinations,
  search_across_all_fields, analyze_with_openai_batch, validate_and_report_data_quality,
  get_comprehensive_server_stats
- OpenAI integration via AsyncOpenAI client with proper error handling
- Proper async/await and full feature set from original implementation

Run:
python mcp_server_full_openai.py
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin

import aiofiles
import aiohttp
import requests
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion

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
    "Summit Open Source MCP (HTTP) with OpenAI",
    stateless_http=True,
    host="0.0.0.0",
    port=9000,
    path="/mcp"
)

# ----------------------------------------------------------------------------
# Configuration & Metadata Database
# ----------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file with fallbacks."""
    config_path = Path("config.json")
    
    # Default configuration
    default_config = {
        "output_dir": Path("summit_data"),
        "database_path": Path("summit_data/metadata.db"),
        "enable_openai": True,
        "max_concurrent_downloads": 3,
        "max_file_size_mb": 50,
        "openai": {
            "api_key": None,
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
            "timeout": 60,
            "enabled": True,
            "max_retries": 3,
            "retry_delay": 1.0
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Merge file config with defaults
            if "openai" in file_config:
                default_config["openai"].update(file_config["openai"])
            
            if "server" in file_config:
                if "output_dir" in file_config["server"]:
                    default_config["output_dir"] = Path(file_config["server"]["output_dir"])
                    default_config["database_path"] = Path(file_config["server"]["output_dir"]) / "metadata.db"
                if "max_file_size_mb" in file_config["server"]:
                    default_config["max_file_size_mb"] = file_config["server"]["max_file_size_mb"]
                if "max_concurrent" in file_config.get("summit_api", {}):
                    default_config["max_concurrent_downloads"] = file_config["summit_api"]["max_concurrent"]
            
            # Enable OpenAI if API key is provided and enabled
            default_config["enable_openai"] = (
                default_config["openai"]["enabled"] and 
                default_config["openai"]["api_key"] and 
                default_config["openai"]["api_key"] != "your-openai-api-key-here"
            )
            
            logger.info(f"Configuration loaded from {config_path}")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Config file {config_path} not found, using defaults")
        # Fallback to environment variable for API key
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key:
            default_config["openai"]["api_key"] = env_api_key
            default_config["enable_openai"] = default_config["openai"]["enabled"]
            logger.info("Using OpenAI API key from environment variable")
    
    return default_config

CONFIG = load_config()

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
            """)

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
            """)

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
                    CONFIG["enable_openai"]
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
# OpenAI Integration
# ----------------------------------------------------------------------------

class OpenAIAnalyzer:
    """Wrapper around the OpenAI Chat Completions API with proper error handling and retry logic."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7, 
                 max_tokens: int = 2000, timeout: int = 60, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout
        )

    async def analyze_metadata(self, metadata: Dict, analysis_type: str = "summary") -> Dict:
        """Analyze metadata using OpenAI Chat Completions API."""
        
        # Build prompt similar to original Ollama implementation
        title = metadata.get("title", [{}])[0].get("value", "Untitled")
        abstract = metadata.get("body", [{}])[0].get("summary", "")
        creators = ", ".join(a.get("value", "") for a in metadata.get("creator", []))
        
        system_prompt = f"""You are a professional metadata analyst. Using ONLY the JSON fields provided, produce a concise {analysis_type} of the item. Do not speculate or add information not present in the metadata."""
        
        user_prompt = f"""TITLE: {title}
CREATORS: {creators}
ABSTRACT: {abstract}

--- FULL METADATA JSON ---
{json.dumps(metadata, indent=2)}
--- END JSON ---

Please provide a {analysis_type} analysis of this metadata."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Implement retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                analysis_text = response.choices[0].message.content
                
                return {
                    "analysis": analysis_text,
                    "type": analysis_type,
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                }
                
            except OpenAIError as e:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return {
                        "analysis": f"Analysis failed after {self.max_retries} attempts. Error: {str(e)}",
                        "type": analysis_type,
                        "error": str(e),
                        "model": self.model
                    }
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI analysis: {e}")
                return {
                    "analysis": f"Analysis failed due to unexpected error: {str(e)}",
                    "type": analysis_type,
                    "error": str(e),
                    "model": self.model
                }

# Initialize OpenAI analyzer
openai_analyzer = None

if CONFIG["enable_openai"] and CONFIG["openai"]["api_key"]:
    try:
        openai_analyzer = OpenAIAnalyzer(
            api_key=CONFIG["openai"]["api_key"],
            model=CONFIG["openai"]["model"],
            temperature=CONFIG["openai"]["temperature"],
            max_tokens=CONFIG["openai"]["max_tokens"],
            timeout=CONFIG["openai"]["timeout"],
            max_retries=CONFIG["openai"]["max_retries"],
            retry_delay=CONFIG["openai"]["retry_delay"]
        )
        logger.info(f"OpenAI analyzer initialized with model: {CONFIG['openai']['model']}")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI analyzer: {e}")
        openai_analyzer = None
        CONFIG["enable_openai"] = False
else:
    if not CONFIG["openai"]["api_key"]:
        logger.info("OpenAI API key not provided. OpenAI features disabled.")
    else:
        logger.info("OpenAI integration disabled in configuration.")
    CONFIG["enable_openai"] = False

# ----------------------------------------------------------------------------
# MCP Tools
# ----------------------------------------------------------------------------

@mcp.tool()
async def fetch_summit_nodes(updated_after: str, max_nodes: int = 100) -> Dict[str, Any]:
    """Fetch Summit nodes with pagination support.
    
    Args:
        updated_after: ISO date string like '2024-01-01' to fetch nodes updated after this date
        max_nodes: Maximum number of nodes to return (default: 100)
    
    Returns:
        Dictionary with 'nodes' list and 'updated_after' timestamp
    """
    async with SummitAPIClient() as client:
        nodes = await client.fetch_paginated_nodes(updated_after, (max_nodes + 19) // 20)
        return {"nodes": nodes[:max_nodes], "updated_after": updated_after}

@mcp.tool()
async def download_complete_node_data(node_id: str, include_files: bool = True, 
                                    perform_analysis: bool = False, max_file_size_mb: int = 50) -> Dict[str, Any]:
    """Download complete node metadata and optionally files with OpenAI analysis.
    
    Args:
        node_id: Summit node ID (e.g. '17', '1234')
        include_files: Whether to download associated files (default: True)
        perform_analysis: Whether to perform OpenAI analysis on metadata (default: False)
        max_file_size_mb: Maximum file size in MB to download (default: 50)
        
    Returns:
        Dictionary with node_id, fields categorization, downloaded files, and optional analysis
    """
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
        if perform_analysis and openai_analyzer:
            analysis = await openai_analyzer.analyze_metadata(detail, "full")

        return {"node_id": node_id, "fields": cats, "files": files, "analysis": analysis}

@mcp.tool()
async def extract_all_field_combinations(node_id: str, combination_type: str = "all") -> Dict[str, Any]:
    """Extract all field path combinations from a Summit node's metadata.
    
    Args:
        node_id: Summit node ID to analyze (e.g. '17', '1234')
        combination_type: Type of field extraction - 'all', 'metadata', or 'content' (default: 'all')
        
    Returns:
        Dictionary with 'paths' containing all field paths and their values
    """
    detail = requests.get(f"https://summit.sfu.ca/node/{node_id}?_format=json").json()
    paths = CompleteFieldProcessor.extract_all_paths(detail)
    return {"paths": paths}

@mcp.tool()
async def search_across_all_fields(query: str, search_type: str = "content", 
                                 category_filter: Optional[str] = None, 
                                 content_type_filter: Optional[str] = None) -> Dict[str, Any]:
    """Search across all stored field data in the database.
    
    Args:
        query: Search query string (e.g. 'climate change', 'machine learning')
        search_type: Type of search - 'content', 'metadata', or 'full' (default: 'content') 
        category_filter: Optional filter by field category - 'string', 'number', or 'other'
        content_type_filter: Optional filter by content type (e.g. 'article', 'dataset')
        
    Returns:
        Dictionary with 'results' array containing matching nid, field_path, and field_value tuples
    """
    with sqlite3.connect(CONFIG["database_path"]) as conn:
        sql = "SELECT nid, field_path, field_value FROM field_paths WHERE field_value LIKE ?"
        params = [f"%{query}%"]
        
        if category_filter:
            sql += " AND category=?"
            params.append(category_filter)
        if content_type_filter:
            sql += " AND nid IN (SELECT nid FROM nodes WHERE content_type=?)"
            params.append(content_type_filter)
            
        rows = conn.execute(sql, params).fetchall()
        return {"results": rows}

@mcp.tool()
async def analyze_with_openai_batch(node_ids: List[str], analysis_types: List[str] = None) -> Dict[str, Any]:
    """Perform batch OpenAI analysis on multiple Summit nodes.
    
    Args:
        node_ids: List of Summit node IDs to analyze (e.g. ['17', '1234', '5678'])
        analysis_types: List of analysis types to perform - 'summary', 'full', 'keywords' (default: ['full'])
        
    Returns:
        Dictionary mapping each node_id to its analysis results by type
    """
    if not openai_analyzer:
        return {"error": "OpenAI analyzer not available"}
        
    types = analysis_types or ["full"]
    out: Dict[str, Any] = {}
    
    for nid in node_ids:
        meta = metadata_db.get_metadata(nid)
        res: Dict[str, Any] = {}
        for t in types:
            res[t] = await openai_analyzer.analyze_metadata(meta, t)
        out[nid] = res
    
    return out

@mcp.tool()
def validate_and_report_data_quality(node_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate and report data quality issues for Summit nodes.
    
    Args:
        node_id: Optional specific node ID to validate (e.g. '17'). If None, validates all nodes.
        
    Returns:
        Dictionary with 'errors' array containing validation issues found
    """
    errs: List[Dict[str, str]] = []
    with sqlite3.connect(CONFIG["database_path"]) as conn:
        if node_id:
            cursor = conn.execute("SELECT metadata_json FROM nodes WHERE nid=?", (node_id,))
        else:
            cursor = conn.execute("SELECT nid, metadata_json FROM nodes")
            
        for row in cursor.fetchall():
            if node_id:
                meta = json.loads(row[0])
                nid = node_id
            else:
                nid, raw = row
                meta = json.loads(raw)
                
            if not meta.get("title"):
                errs.append({"nid": nid, "issue": "Missing title"})
                
    return {"errors": errs}

@mcp.tool()
async def download_file(node_id: str, max_size_mb: int = CONFIG["max_file_size_mb"], 
                       include_json: bool = True) -> Dict[str, Any]:
    """Download a Summit node's primary PDF and optionally save metadata JSON.
    
    Args:
        node_id: Summit node ID to download files from (e.g. '17', '1234')
        max_size_mb: Maximum file size in MB before aborting download (default: 50)
        include_json: Whether to save metadata JSON file alongside PDF (default: True)
        
    Returns:
        Dictionary with download details including file paths and sizes
    """
    async with SummitAPIClient() as client:
        detail = await client.fetch_node_detail(node_id)
        
        json_saved_to = None
        if include_json:
            json_path = CONFIG["output_dir"] / str(node_id) / f"{node_id}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
            json_saved_to = str(json_path)

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
    """Get comprehensive server statistics and status information.
    
    Returns:
        Dictionary with server statistics including:
        - timestamp: Current server time
        - nodes_by_type: Count of nodes by content type  
        - fields_by_category: Count of field paths by category
        - total_nodes: Total number of nodes stored
        - total_field_paths: Total number of field paths stored
        - storage_mb: Total storage used in MB
        - openai_enabled: Whether OpenAI integration is active
        - openai_model: Current OpenAI model in use
    """
    stats: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat() + "Z"}
    stats.update(metadata_db.get_statistics())
    
    size = sum(f.stat().st_size for f in CONFIG["output_dir"].rglob("*") if f.is_file())
    stats["storage_mb"] = round(size / 1024**2, 2)
    
    # Add OpenAI configuration status
    stats["openai_enabled"] = CONFIG["enable_openai"] and openai_analyzer is not None
    stats["openai_model"] = CONFIG["openai"]["model"] if openai_analyzer else None
    
    return stats

# ----------------------------------------------------------------------------
# Run Server
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--disable-openai', action='store_true')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()

    # Override config if command line arguments provided
    if args.disable_openai:
        CONFIG['enable_openai'] = False
        logger.info("OpenAI integration disabled via command line")

    mcp.host = args.host
    mcp.port = args.port
    
    # Print configuration status
    logger.info(f"Starting Summit MCP Server with OpenAI integration on {args.host}:{args.port}")
    logger.info(f"OpenAI enabled: {CONFIG['enable_openai']}")
    if CONFIG['enable_openai']:
        logger.info(f"OpenAI model: {CONFIG['openai']['model']}")
    logger.info(f"Output directory: {CONFIG['output_dir']}")
    logger.info(f"Database path: {CONFIG['database_path']}")
    
    if not CONFIG['enable_openai']:
        logger.info("To enable OpenAI features:")
        logger.info("1. Edit config.json and add your OpenAI API key")
        logger.info("2. Set 'enabled': true in the openai section")
        logger.info("3. Restart the server")
    
    mcp.run(transport='streamable-http')
