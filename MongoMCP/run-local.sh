#!/bin/bash

# MongoDB MCP Server - Local Standalone Startup Script (Bash/Linux/macOS)
# This script loads environment variables and starts the MCP server in local mode
# BB - fastapi run mongo_mcp.py --port 8001
# USE_LOCAL_MODE=true
# Manually set MONGO_PASSWORD=<password> in your environment.
# PATH=/Users/brady.byrd/Documents/mongodb/dev/mongo-examples/mcpclient/bin
# !IMPORTANT! - you must have you AWS session set in your environment for this to work.
#   use>  aws sso login

set -e  # Exit on error

echo "=========================================="
echo "MongoDB MCP Server - Local Mode"
echo "=========================================="

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "ERROR: .env.local file not found!"
    echo ""
    echo "Please create .env.local from the template:"
    echo "  cp .env.local.example .env.local"
    echo ""
    echo "Then edit .env.local with your MongoDB credentials and configuration."
    exit 1
fi

# Load environment variables from .env.local
echo "Loading environment variables from .env.local..."
echo export $(grep -v '^#' .env.local | grep -v '^$' | xargs)
export $(grep -v '^#' .env.local | grep -v '^$' | xargs)

# Verify required environment variables
if [ -z "$MCP_TOOL_NAME" ]; then
    echo "ERROR: MCP_TOOL_NAME is not set in .env.local"
    exit 1
fi

if [ -z "$MONGO_CONNECTION_STRING" ] && { [ -z "$MONGO_USERNAME" ] || [ -z "$MONGO_PASSWORD" ] || [ -z "$MONGO_URI" ]; }; then
    echo "ERROR: MongoDB credentials not configured in .env.local"
    echo "Please set either MONGO_CONNECTION_STRING or (MONGO_USERNAME, MONGO_PASSWORD, MONGO_URI)"
    exit 1
fi

echo "Configuration loaded successfully!"
echo "Tool Name: $MCP_TOOL_NAME"
echo "Config DB: ${MCP_CONFIG_DB:-mcp_config}"
echo "Config Collection: ${MCP_CONFIG_COL:-mcp_tools}"
echo ""

# Parse command line arguments
HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${SERVER_PORT:-8000}"
TRANSPORT="http"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --sse)
            TRANSPORT="sse"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST      Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT      Port to bind to (default: 8000)"
            echo "  --sse            Use SSE transport instead of HTTP"
            echo "  --help           Show this help message"
            echo ""
            echo "Environment variables are loaded from .env.local"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Start the server
echo "Starting MongoDB MCP Server..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Transport: $TRANSPORT"
echo "=========================================="
echo ""

if [ "$TRANSPORT" = "sse" ]; then
    fastapi run mongo_mcp.py --port $PORT
else
    fastapi run mongo_mcp.py --port $PORT
fi

