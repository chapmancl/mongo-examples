# Use a base image with the desired runtime environment
FROM packages.lmig.com/image-lm-base/python3-ubuntu-mutable:latest

#RUN apt-get update && apt-get install -y curl

ARG CACHE_BUST=675843
# Set the working directory inside the container
WORKDIR /app

# Copy the application dependencies file to the container
COPY requirements.txt ./
COPY agent_instructions.md .

# Install the application dependencies
RUN --mount=type=secret,id=netrc,target=/root/.netrc,uid=1000 pip install -r requirements.txt
# Copy the application code to the container
COPY mongomcp ./mongomcp
COPY tools ./tools
COPY webui ./webui
COPY certs ./certs
COPY *.py ./

#RUN --mount=type=secret,id=netrc,target=/root/.netrc,uid=1000 pip install --no-cache-dir ./mongomcp

RUN echo `pwd`

RUN echo $CACHE_BUST && ls -la

# Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV FASTMCP_STATELESS_HTTP=1
ENV IS_LOCAL=true
ENV USE_LOCAL_MODE=true
# ============================================================================
# REQUIRED: Tool Configuration
# ============================================================================
# Name of the MCP tool to load from MongoDB configuration
# This must match a "Name" field in your mcp_config.mcp_tools collection
ENV MCP_TOOL_NAME=gemSearch

# Embedding
#ENV QUERY_EMBEDDING_MODEL_ID=openai-text-embedding-3-small-v1
#ENV EMBEDDING_MODEL_ID=openai-text-embedding-3-small-v1
ENV QUERY_EMBEDDING_MODEL_ID=voyage-4-lite
ENV EMBEDDING_MODEL_ID=voyage-4
#ENV QUERY_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
#ENV EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0


# ============================================================================
# REQUIRED: MongoDB Connection (Choose ONE option)
# ============================================================================
ENV MONGO_URI=gem-exposure-repository-mongo-db-test-pl-0.z6v9s.mongodb.net

# ============================================================================
# OPTIONAL: MongoDB Configuration Database
# ============================================================================
# Database and collection where MCP tool configurations are stored
ENV MCP_CONFIG_DB=ai_config
ENV MCP_CONFIG_COL=mcp_tools
ENV WEBSERVER_PORT=8001
ENV MONGO_MCP_ROOT=https://gem-mongodb-mcpserver-development.lmig.com

# ============================================================================
# OPTIONAL: AWS Configuration (for Bedrock LLM)
# ============================================================================
# AWS region for Bedrock service
ENV AWS_REGION=us-east-1
# AWS credentials (if not using AWS CLI profile)
#ENV AWS_BEARER_TOKEN_BEDROCK=""
# Set the command to run the application (replace with your command)
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--chdir", "webui", "app:app"]
#CMD ["fastapi", "run", "mongo_mcp.py"]
RUN echo $CACHE_BUST && env
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FASTMCP_STATELESS_HTTP=1
ENV FASTMCP_LOG_LEVEL=INFO

EXPOSE 8000

CMD ["-m", "fastapi", "run", "mongo_mcp.py"]

