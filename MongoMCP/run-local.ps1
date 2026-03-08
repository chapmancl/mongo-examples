# MongoDB MCP Server - Local Standalone Startup Script (PowerShell/Windows)
# This script loads environment variables and starts the MCP server in local mode

param(
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$SSE,
    [switch]$Help
)

function Show-Help {
    Write-Host "MongoDB MCP Server - Local Mode Startup Script"
    Write-Host ""
    Write-Host "Usage: .\run-local.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Host <host>     Host to bind to (default: 0.0.0.0)"
    Write-Host "  -Port <port>     Port to bind to (default: 8000)"
    Write-Host "  -SSE             Use SSE transport instead of HTTP"
    Write-Host "  -Help            Show this help message"
    Write-Host ""
    Write-Host "Environment variables are loaded from .env.local"
    exit 0
}

if ($Help) {
    Show-Help
}

Write-Host "=========================================="
Write-Host "MongoDB MCP Server - Local Mode"
Write-Host "=========================================="

# Check if .env.local exists
if (-not (Test-Path ".env.local")) {
    Write-Host "ERROR: .env.local file not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please create .env.local from the template:"
    Write-Host "  Copy-Item .env.local.example .env.local"
    Write-Host ""
    Write-Host "Then edit .env.local with your MongoDB credentials and configuration."
    exit 1
}

# Load environment variables from .env.local
Write-Host "Loading environment variables from .env.local..."
Get-Content ".env.local" | ForEach-Object {
    $line = $_.Trim()
    # Skip comments and empty lines
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split '=', 2
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Verify required environment variables
$MCP_TOOL_NAME = [Environment]::GetEnvironmentVariable("MCP_TOOL_NAME", "Process")
$MONGO_CONNECTION_STRING = [Environment]::GetEnvironmentVariable("MONGO_CONNECTION_STRING", "Process")
$MONGO_USERNAME = [Environment]::GetEnvironmentVariable("MONGO_USERNAME", "Process")
$MONGO_PASSWORD = [Environment]::GetEnvironmentVariable("MONGO_PASSWORD", "Process")
$MONGO_URI = [Environment]::GetEnvironmentVariable("MONGO_URI", "Process")

if (-not $MCP_TOOL_NAME) {
    Write-Host "ERROR: MCP_TOOL_NAME is not set in .env.local" -ForegroundColor Red
    exit 1
}

if (-not $MONGO_CONNECTION_STRING -and (-not $MONGO_USERNAME -or -not $MONGO_PASSWORD -or -not $MONGO_URI)) {
    Write-Host "ERROR: MongoDB credentials not configured in .env.local" -ForegroundColor Red
    Write-Host "Please set either MONGO_CONNECTION_STRING or (MONGO_USERNAME, MONGO_PASSWORD, MONGO_URI)"
    exit 1
}

Write-Host "Configuration loaded successfully!" -ForegroundColor Green
Write-Host "Tool Name: $MCP_TOOL_NAME"
$MCP_CONFIG_DB = [Environment]::GetEnvironmentVariable("MCP_CONFIG_DB", "Process")
$MCP_CONFIG_COL = [Environment]::GetEnvironmentVariable("MCP_CONFIG_COL", "Process")
Write-Host "Config DB: $(if ($MCP_CONFIG_DB) { $MCP_CONFIG_DB } else { 'mcp_config' })"
Write-Host "Config Collection: $(if ($MCP_CONFIG_COL) { $MCP_CONFIG_COL } else { 'mcp_tools' })"
Write-Host ""

# Override from environment if set
$SERVER_HOST = [Environment]::GetEnvironmentVariable("SERVER_HOST", "Process")
$SERVER_PORT = [Environment]::GetEnvironmentVariable("SERVER_PORT", "Process")

if ($SERVER_HOST) { $Host = $SERVER_HOST }
if ($SERVER_PORT) { $Port = [int]$SERVER_PORT }

# Determine transport
$Transport = if ($SSE) { "sse" } else { "http" }

# Start the server
Write-Host "Starting MongoDB MCP Server..."
Write-Host "Host: $Host"
Write-Host "Port: $Port"
Write-Host "Transport: $Transport"
Write-Host "=========================================="
Write-Host ""

# Build command arguments
$args = @("mongo_mcp.py", "--host", $Host, "--port", $Port, "--local")
if ($SSE) {
    $args += @("--transport", "sse")
}

# Run Python
& python $args

