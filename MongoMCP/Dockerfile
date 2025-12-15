#FROM --platform=linux/arm64 python:3.11-slim
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY MongoMCP/ MongoMCP/
COPY mongo_mcp.py .
COPY health.py .

# Make the script executable
RUN chmod +x mongo_mcp.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the MCP server
CMD ["fastapi", "run", "mongo_mcp.py"]
