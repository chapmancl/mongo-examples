#FROM --platform=linux/arm64 python:3.11-slim
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY MongoDBVectorServer.py .
COPY mongo_mcp.py .
COPY settings_aws.py .
COPY health.py .

# Make the script executable
RUN chmod +x mongo_mcp.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the MCP server
CMD ["python", "mongo_mcp.py"]
#CMD ["fastmcp", "run", "mongo_mcp.py", "--transport", "http","--port","8000"]