# MongoDB Vector Search MCP Server

A custom Model Context Protocol (MCP) server that provides vector search capabilities for MongoDB using the `$vectorSearch` aggregation pipeline. This server connects to MongoDB Atlas and enables semantic search operations on vector embeddings.

## Features

- **Vector Search**: Perform semantic similarity search using MongoDB's `$vectorSearch` aggregation pipeline with AI embeddings
- **Text Search**: Full-text search using MongoDB's `$search` aggregation pipeline with keyword matching
- **Unique Values Discovery**: Get unique values for any field to discover available filter options
- **Custom Aggregation Queries**: Execute complex MongoDB aggregation pipelines for advanced data analysis
- **Collection Info**: Get comprehensive metadata about the MongoDB collection, indexes, and search capabilities
- **Configurable**: Uses `settings_aws.py` for MongoDB connection configuration through AWS Secrets Manager

## Prerequisites

- Python 3.8+
- MongoDB Atlas cluster with vector search index configured
- MCP client 

## How to Run the MCP service
1. Setup Mongo (see [MongoDB Configuration](#mongodb-configuration) below)
2. Setup your python environment (see [Python Virtual Environment Setup](#python-virtual-environment-setup))
3. Install requirements (see [Installation](#installation))
4. Run fastmcp (see [FastMCP Deployment](#fastmcp-deployment))

## MongoDB Configuration

## Data Structure

This server is designed to work with the [MongoDB Atlas Sample Airbnb Dataset](https://www.mongodb.com/docs/atlas/sample-data/sample-airbnb/). The server expects documents in your MongoDB collection to have the following structure:

```json
{
  "_id": "...",
  "name": "Property Name",
  "description": "Property description",
  "property_type": "Apartment",
  "room_type": "Entire home/apt",
  "accommodates": 4,
  "beds": 2,
  "bedrooms": 1,
  "price": "$100.00",
  "embedding": [0.1, 0.2, 0.3, ...],
  "neighborhood_overview": "Great location...",
  "address": {
    "country_code": "US",
    "market": "New York",
    "suburb": "Brooklyn"
  }
}
```


1. **Load the Sample Dataset**: In your MongoDB Atlas cluster, load the sample datasets which includes the `sample_airbnb.listingsAndReviews` collection.

2. **Add Vector Embeddings**: The sample dataset doesn't include vector embeddings by default. You'll need to generate 1024-dimensional embeddings for the text fields (name, description, neighborhood_overview) and add them as an `embedding` field to each document.

3. **Create Vector Search Index**: Configure the `listing_vector_index` vector search index on the `embedding` field as shown in the MongoDB Configuration section above.


### Vector Search Index

Your MongoDB collection should have a vector search index named `listing_vector_index` configured. Index definition:

```json
{
  "fields": [
    {
      "numDimensions": 1024,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "address.country_code",
      "type": "filter"
    },
    {
      "path": "address.market",
      "type": "filter"
    },
    {
      "path": "beds",
      "type": "filter"
    },
    {
      "path": "bedrooms",
      "type": "filter"
    },
    {
      "path": "address.suburb",
      "type": "filter"
    }
  ]
}
```

### Text Search Index

For text search functionality, ensure you have a text search index named `search_index`:

```json
{
  "analyzer": "lucene.english",
  "searchAnalyzer": "lucene.english",
  "mappings": {
    "dynamic": false,
    "fields": {
      "amenities": [
        {
          "type": "stringFacet"
        },
        {
          "type": "token"
        }
      ],
      "beds": [
        {
          "type": "numberFacet"
        },
        {
          "type": "number"
        }
      ],
      "description": [
        {
          "type": "stringFacet"
        },
        {
          "type": "token"
        }
      ],
      "name": {
        "analyzer": "lucene.english",
        "foldDiacritics": false,
        "maxGrams": 7,
        "minGrams": 3,
        "type": "autocomplete"
      },
      "property_type": [
        {
          "type": "stringFacet"
        },
        {
          "type": "token"
        }
      ],
      "summary": [
        {
          "type": "stringFacet"
        },
        {
          "type": "token"
        }
      ]
    }
  }
}
```

## Python Virtual Environment Setup

It's recommended to use a Python virtual environment to isolate dependencies and avoid conflicts with other projects.

1. **Create a virtual environment**:
   ```bash
   python -m venv mongodb-mcp-env
   ```

2. **Activate the virtual environment**:
   
   **On Linux/macOS:**
   ```bash
   source mongodb-mcp-env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS Environment Variables**:
   Set the following environment variables for AWS Secrets Manager integration:
   ```bash
   export AWS_REGION=us-east-2
   export MONGO_CREDS=your-secrets-manager-secret-name
   export MONGO_DB=your_database
   export MONGO_COL=your_collection
   ```

   The `MONGO_CREDS` secret in AWS Secrets Manager should contain:
   ```json
   {
     "username": "your_mongodb_username",
     "password": "your_mongodb_password", 
     "uri": "cluster.mongodb.net"
   }
   ```

## FastMCP Deployment

This server can be deployed using [FastMCP](https://gofastmcp.com/) for enhanced deployment options and multiple transport capabilities.
For more FastMCP deployment options, see the [FastMCP documentation](https://gofastmcp.com/deployment/running-server).

**Default HTTP Transport (for production/containers):**
```bash
python mongo_mcp.py
```

The server will start with HTTP transport on port 8000 at `http://localhost:8000/mcp/`.

**For Local IDE Integration (Cline, Copilot, etc.):**
```bash
fastmcp run mongo_mcp.py --transport sse --port 8001
```

This starts the server with SSE (Server-Sent Events) transport for local development and IDE integration.

### Docker with FastMCP

Build the Docker image for the MCP server:

```bash
docker build -t mongodb-vector-mcp .
```

### Pushing to Amazon ECR

1. **Create an ECR repository** (if it doesn't exist):
   ```bash
   aws ecr create-repository --repository-name mongodb-vector-mcp --region us-east-2
   ```

2. **Get the login token and authenticate Docker to ECR**:
   ```bash
   aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-2.amazonaws.com
   ```

3. **Tag the image for ECR**:
   ```bash
   docker tag mongodb-vector-mcp:latest <account-id>.dkr.ecr.us-east-2.amazonaws.com/mongodb-vector-mcp:latest
   ```

4. **Push the image to ECR**:
   ```bash
   docker push <account-id>.dkr.ecr.us-east-2.amazonaws.com/mongodb-vector-mcp:latest
   ```

Replace `<account-id>` with your AWS account ID.

You can run the server in Docker with FastMCP HTTP transport by with the Dockerfile CMD:

```dockerfile
CMD ["python", "mongo_mcp.py"]
```

Then run the container locally with port mapping:

```bash
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-2 \
  -e MONGO_CREDS=your-secret-name \
  -e MONGO_DB=your_database \
  -e MONGO_COL=your_collection \
  mongodb-vector-mcp:latest
```


### Available Tools

#### 1. `vector_search`
Perform semantic vector similarity search on MongoDB collection using AI embeddings.

**Parameters:**
- `query_text` (required): Natural language query describing desired property characteristics
- `limit` (optional): Maximum number of results (default: 10, max: 50)
- `num_candidates` (optional): Number of candidates to consider (default: 100, max: 1000)
- `filters` (optional): List of filters to narrow search results (e.g., [["beds", 2], ["address.country_code", "US"]])

**Example:**
```json
{
  "query_text": "cozy apartment near Central Park",
  "limit": 5,
  "num_candidates": 50,
  "filters": [["beds", 2], ["address.country_code", "US"]]
}
```

#### 2. `text_search`
Perform traditional keyword-based text search using Atlas Search.

**Parameters:**
- `query_text` (required): Keywords or phrases to search for
- `limit` (optional): Maximum number of results (default: 10, max: 100)

**Example:**
```json
{
  "query_text": "2 bedroom apartment WiFi kitchen",
  "limit": 10
}
```

#### 3. `get_unique_values`
Get unique values for a specific field to discover available filter options.

**Parameters:**
- `field` (required): Field name to get unique values for (supports dot notation)

**Example:**
```json
{
  "field": "address.market"
}
```

#### 4. `aggregate_query`
Execute custom MongoDB aggregation pipeline queries for complex data analysis.

**Parameters:**
- `pipeline` (required): List of aggregation stage dictionaries
- `limit` (optional): Optional limit for results (default: None, max: 1000)

**Example:**
```json
{
  "pipeline": [
    {"$group": {"_id": "$property_type", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
  ],
  "limit": 10
}
```

#### 5. `get_collection_info`
Get comprehensive information about the MongoDB collection, database statistics, and search capabilities.



## Integration with MCP Clients

### Amazon Bedrock Agents

This MCP server is designed to work seamlessly with Amazon Bedrock Agents using the Inline Agent SDK. Based on the AWS blog post about [MCP servers with Amazon Bedrock Agents](https://aws.amazon.com/blogs/machine-learning/harness-the-power-of-mcp-servers-with-amazon-bedrock-agents/), you can integrate this server as follows:

1. **Build the Docker image**:
   ```bash
   docker build -t mongodb-vector-mcp .
   ```

2. **Use with Amazon Bedrock Inline Agent**:
   ```python
   from mcp.client.stdio import MCPStdio, StdioServerParameters
   from inline_agent import InlineAgent, ActionGroup
   
   # Configure MCP server parameters
   mongodb_server_params = StdioServerParameters(
       command="docker",
       args=[
           "run", "-i", "--rm",
           "-e", "AWS_REGION",
           "-e", "MONGO_CREDS", 
           "-e", "MONGO_DB",
           "-e", "MONGO_COL",
           "mongodb-vector-mcp:latest"
       ],
       env={
           "AWS_REGION": "us-east-2",
           "MONGO_CREDS": "your-secrets-manager-secret-name",
           "MONGO_DB": "demo1",
           "MONGO_COL": "sample_airbnb"
       }
   )
   
   # Create MCP client and agent
   mongodb_mcp_client = await MCPStdio.create(server_params=mongodb_server_params)
   
   agent = InlineAgent(
       foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
       instruction="You are a helpful assistant for MongoDB vector search operations.",
       agent_name="MongoDBVectorSearchAgent",
       action_groups=[
           ActionGroup(
               name="MongoDBVectorSearchActionGroup",
               mcp_clients=[mongodb_mcp_client]
           )
       ]
   )
   
   # Use the agent
   response = await agent.invoke(
       input_text="Find similar properties to luxury apartments"
   )
   ```

### Claude Desktop

Add to your MCP configuration file:

```json
{
  "mcpServers": {
    "mongodb-vector": {
      "command": "python",
      "args": ["/path/to/your/mcp.py"],
      "env": {}
    }
  }
}
```

### Cline (VS Code Extension)

Cline is a popular VS Code extension that supports MCP servers. To integrate this MongoDB vector search server with Cline:

2. **Configure Cline MCP Settings**:
   Open VS Code settings (Ctrl/Cmd + ,) and search for "cline mcp" or edit your VS Code `cline_mcp_settings.json`:

   ```json
   {
     "cline.mcpServers": {
       "mongodb-vector-search": {
         "command": "fastmcp",
         "args": ["run", "mongo_mcp.py", "--transport", "sse", "--port", "8001"],
         "cwd": "/path/to/your/mongodb-mcp-project",
         "env": {
           "AWS_REGION": "us-east-2",
           "MONGO_CREDS": "your-secrets-manager-secret-name",
           "MONGO_DB": "your_database",
           "MONGO_COL": "your_collection"
         }
       }
     }
   }
   ```

3. **Alternative Configuration** (if running the server separately):
   If you prefer to run the server manually, start it with:
   ```bash
   fastmcp run mongo_mcp.py --transport sse --port 8001
   ```
   
   Then configure Cline to connect to the running server:
   ```json
   {
     "cline.mcpServers": {
       "mongodb-vector-search": {
         "url": "http://localhost:8001/sse"
       }
     }
   }
   ```

### Other MCP Clients

The server follows the standard MCP protocol and should work with any MCP-compatible client. For clients that support HTTP transport, connect to `http://localhost:8000/mcp/` when running with the default configuration.


## Troubleshooting

**Connection Issues**: Verify your MongoDB URI and network connectivity
**Index Errors**: Ensure your vector search index is properly configured
**Vector Dimension Mismatch**: Check that your query vector dimensions match the index configuration
**AWS Authentication Issues**: If you encounter the error `Invalid type for parameter SecretId`, this typically indicates an AWS authentication or configuration issue.


## Contributing

Feel free to submit issues and enhancement requests!
