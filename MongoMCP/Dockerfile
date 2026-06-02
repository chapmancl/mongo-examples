#FROM --platform=linux/arm64 python:3.11-slim
#FROM python:3.13-slim
FROM mcp-base:latest

# Set working directory
WORKDIR /app

# ── Dependency layer (cached until requirements files change) ──────────────────
#COPY requirements.txt .
ARG CACHEBUST=1
COPY mongomcp/ mongomcp/
RUN pip install --no-cache ./mongomcp
#RUN pip install --no-cache-dir -r requirements.txt

# ── Application code (changes frequently, doesn't bust dep cache) ─────────────
COPY agent_instructions.md .
COPY AWS_settings.py .
COPY mongo_mcp.py .

# Make the script executable
#RUN chmod +x mongo_mcp.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FASTMCP_STATELESS_HTTP=1
ENV FASTMCP_LOG_LEVEL=INFO

EXPOSE 8000

# Run the MCP server
CMD ["fastapi", "run", "mongo_mcp.py"]
