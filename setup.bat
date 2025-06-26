@echo off
setlocal enabledelayedexpansion

echo ================================================
echo Summit Open-Source MCP Server Setup
echo Complete Open-Source Implementation
echo ================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo ✓ Python found
python --version

:: Create virtual environment
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install core dependencies
echo Installing core dependencies...
pip install aiofiles aiohttp requests pathlib
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)

:: Install MCP SDK
echo Installing official MCP Python SDK...
pip install mcp
if errorlevel 1 (
    echo ERROR: Failed to install MCP SDK
    pause
    exit /b 1
)

echo ✓ All Python dependencies installed

:: Check Ollama installation
echo Checking for Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama not found
    echo.
    echo To enable local LLM analysis:
    echo 1. Download Ollama from https://ollama.com
    echo 2. Install Ollama
    echo 3. Run: ollama pull llama3.2:latest
    echo.
    echo The server will work without Ollama but analysis features will be disabled
    echo.
    set OLLAMA_MISSING=1
) else (
    echo ✓ Ollama found
    ollama --version
    
    :: Check if required model exists
    echo Checking for llama3.2:latest model...
    ollama list | findstr "llama3.2:latest" >nul
    if errorlevel 1 (
        echo Downloading llama3.2:latest model...
        echo This may take several minutes depending on your internet connection...
        echo.
        ollama pull llama3.2:latest
        if errorlevel 1 (
            echo WARNING: Failed to download model
            echo Analysis features will be limited
            set OLLAMA_LIMITED=1
        ) else (
            echo ✓ Model downloaded successfully
        )
    ) else (
        echo ✓ llama3.2:latest model available
    )
    
    :: Start Ollama service if not running
    echo Checking Ollama service...
    netstat -an | findstr ":11434" >nul
    if errorlevel 1 (
        echo Starting Ollama service...
        start /B ollama serve
        timeout /t 3 /nobreak >nul
        echo ✓ Ollama service started
    ) else (
        echo ✓ Ollama service already running
    )
)

:: Create project directories
echo Creating project directories...
if not exist "summit_data" mkdir summit_data
if not exist "logs" mkdir logs
echo ✓ Directories created

:: Create configuration file
echo Creating configuration file...
(
echo {
echo   "server": {
echo     "name": "Summit Open Source MCP Processor",
echo     "version": "1.0.0",
echo     "output_dir": "summit_data",
echo     "max_file_size_mb": 50
echo   },
echo   "summit_api": {
echo     "base_url": "https://summit.sfu.ca",
echo     "timeout": 30,
echo     "max_concurrent": 3
echo   },
echo   "ollama": {
echo     "host": "localhost:11434",
echo     "model": "llama3.2:latest",
echo     "enabled": %if defined OLLAMA_MISSING (echo false% else echo true%
echo   },
echo   "logging": {
echo     "level": "INFO",
echo     "file": "logs/mcp_server.log"
echo   }
echo }
) > config.json
echo ✓ Configuration file created

:: Test server startup
echo.
echo Testing server components...
python -c "import mcp; print('✓ MCP SDK imported successfully')"
python -c "import aiohttp, aiofiles; print('✓ Async libraries imported')"
python -c "import sqlite3; print('✓ SQLite available')"

:: Create server script if it doesn't exist
if not exist "summit_mcp_server.py" (
    echo ERROR: summit_mcp_server.py not found
    echo Please ensure the server script is in the current directory
    pause
    exit /b 1
)

echo ✓ Server script found

echo.
echo ================================================
echo Setup completed successfully!
echo ================================================
echo.
echo Next steps:
echo.
echo 1. Start the server:
echo    run_server.bat
echo.
echo 2. Test the server:
echo    test_server.bat
echo.
echo 3. View server status:
echo    status.bat
echo.

if defined OLLAMA_MISSING (
    echo NOTE: Ollama is not installed
    echo Install Ollama to enable local LLM analysis features
    echo.
)

if defined OLLAMA_LIMITED (
    echo NOTE: Ollama model download failed
    echo Try running: ollama pull llama3.2:latest
    echo.
)

echo The server is ready to run in open-source mode!
echo.
pause
