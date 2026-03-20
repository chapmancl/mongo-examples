#FROM --platform=linux/arm64 python:3.11-slim
FROM python:3.13-slim
RUN apt-get update && apt-get install -y curl
# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Copy application files
COPY mongomcp/ mongomcp/
COPY AWS_settings.py .
COPY mongo_mcp.py .

RUN pip install --no-cache-dir ./mongomcp
RUN pip install -r requirements.txt

# Make the script executable
#RUN chmod +x mongo_mcp.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FASTMCP_STATELESS_HTTP=1

EXPOSE 8000

# Run the MCP server
CMD ["fastapi", "run", "mongo_mcp.py"]
