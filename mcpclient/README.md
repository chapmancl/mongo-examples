# Airbnb MCP Client

This is a Model Context Protocol (MCP) client that connects to a MongoDB Atlas vector search MCP server and AWS Bedrock to process user queries using Claude LLM with tool support.

## Python Setup Instructions

### Prerequisites
- Python 3.8 or higher
- AWS credentials configured (for Bedrock access)
- Access to a custom MCP server with MongoDB vector search capabilities

### 1. Create Python Virtual Environment

```bash
# Create a new virtual environment
python -m venv .

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Configuration

Before running the application, you need to configure your settings:

1. Copy or create a `settingsairbnb.py` file with your configuration:
   - AWS region settings
   - MongoDB MCP server connection details
   - Bedrock model ID

Example `settingsairbnb.py`:
```python
# AWS Configuration
aws_region = "us-east-1"
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# MCP Server Configuration
mong_mcp = "your-mcp-server-endpoint"
```

### 4. Run the Application

```bash
python airbnb-mcp.py
```

## Usage

Once the application is running, you can:

- Enter questions directly to query Claude with MCP tool support
- Use `clear` command to clear conversation history and tools
- Press Ctrl+C to exit

## Features

- **MCP Tool Integration**: Automatically discovers and uses tools from connected MCP servers
- **AWS Bedrock Integration**: Uses Claude LLM via AWS Bedrock Converse API
- **Conversation History**: Maintains context across multiple queries


## Troubleshooting

- Ensure your AWS credentials are properly configured
- Verify your MCP server is running and accessible
- Make sure you have the necessary AWS Bedrock model access permissions
- Make sure to use the Cross-region inference profile ID for the given model.
https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html?icmpid=docs_bedrock_help_panel_cross_region_inference
