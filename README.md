# MongoDB Examples with AI and Vector Search

This repository contains three complementary projects that demonstrate MongoDB Atlas integration with AI services, vector search capabilities, and Model Context Protocol (MCP) implementations for the MongoDB sample Airbnb dataset.

### [jsonembed/](./jsonembed/)
**Document Embedding and Vectorization**

A Python application that processes MongoDB documents, generates AI embeddings using AWS Bedrock, and stores vector representations for semantic search capabilities.

- Document chunking and text extraction from JSON
- AWS Bedrock Titan embeddings integration
- Batch processing of MongoDB collections
- Vector storage for similarity search

### [MongoMCP/](./MongoMCP/)
**MongoDB MCP Server**

A Model Context Protocol (MCP) server that provides vector search and other capabilities for MongoDB Atlas. This server enables semantic search operations on vector embeddings and integrates with AI agents and tools.

- Vector similarity search using MongoDB `$vectorSearch`
- Text search with Atlas Search
- Custom aggregation queries
- MCP protocol compliance for AI agent integration

### [mcpclient/](./mcpclient/)
**MCP Client with AWS Bedrock Integration**

An interactive client application that connects to MCP servers and uses AWS Bedrock's Claude LLM for intelligent query processing with tool support.

- MCP tool discovery and execution
- AWS Bedrock Claude LLM integration
- Interactive command-line interface
- Conversation history management
- Tool-assisted query processing

## Workflow Overview

1. **Data Preparation** (`jsonembed/`): Process raw MongoDB documents and generate vector embeddings
2. **Search Service** (`searchmcp/`): Deploy MCP server to provide vector search capabilities
3. **AI Interface** (`mcpclient/`): Use intelligent client to query data through natural language

## Prerequisites

- Python 3.8 or higher
- MongoDB Atlas cluster with sample Airbnb dataset
- AWS account with Bedrock access
- Docker (optional, for containerized deployment)

## Quick Start

1. **Set up vector embeddings:**
   ```bash
   cd jsonembed/
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   python embedairbnb.py
   ```

2. **Deploy MCP server:**
   ```bash
   cd searchmcp/
   python -m venv venv
   source venv/bin/activate 
   pip install -r requirements.txt
   fastmcp run mongo_mcp.py --transport http --port 8001
   ```

3. **Run interactive client:**
   ```bash
   cd mcpclient/
   python -m venv venv
   source venv/bin/activate 
   pip install -r requirements.txt
   python airbnb-mcp.py
   ```

## Configuration

Each project requires configuration files (`settings.py` or `settings_aws.py`) with:
- MongoDB Atlas connection details
- AWS credentials and region settings
- Model IDs for Bedrock services

## Contributing

Each project folder contains its own documentation and setup instructions. Please refer to the individual README files for detailed information about each component.

## License

See [LICENSE](./LICENSE) file for details.
