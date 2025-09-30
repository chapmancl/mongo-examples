# Dynamic MongoDB MCP Server

A highly configurable Model Context Protocol (MCP) server that dynamically loads tool configurations from a MongoDB collection. This server enables flexible, database-driven MCP tool generation for various MongoDB collections and use cases.

## Features

- **Dynamic Configuration**: MCP server configuration is dynamically loaded from a MongoDB collection, allowing for flexible tool definitions without code changes
- **Automatic Tool Generation**: Tool information and metadata are dynamically generated based on JSON configuration documents stored in MongoDB
- **Vector Search**: Perform semantic similarity search using MongoDB's `$vectorSearch` aggregation pipeline with AI embeddings
- **Text Search**: Full-text search using MongoDB's `$search` aggregation pipeline with keyword matching
- **Unique Values Discovery**: Get unique values for any field to discover available filter options
- **Custom Aggregation Queries**: Execute complex MongoDB aggregation pipelines for advanced data analysis
- **Collection Info**: Get comprehensive metadata about the MongoDB collection, indexes, and search capabilities
- **Multi-Configuration Support**: Support for multiple MCP server configurations in a single MongoDB collection

## Prerequisites

- Python 3.8+
- MongoDB Atlas cluster with MCP configuration collection
- MongoDB Atlas cluster with target data collection(s) (optionally with vector search index configured)
- MCP client 

## How to Run the MCP service
1. Setup MongoDB with MCP configurations (see [Dynamic Configuration Setup](#dynamic-configuration-setup) below)
2. Setup your python environment (see [Python Virtual Environment Setup](#python-virtual-environment-setup))
3. Install requirements (see [Installation](#installation))
4. Run fastmcp (see [FastMCP Deployment](#fastmcp-deployment))

## Dynamic Configuration Setup

This MCP server dynamically loads its configuration from a MongoDB collection. The configuration defines which tools are available, their parameters, descriptions, and behavior. This allows you to create multiple MCP server configurations for different databases and collections without modifying code.

### Configuration Collection

Create a MongoDB collection to store your MCP configurations (e.g., `mcp_configurations`). Each document in this collection defines a complete MCP server configuration.

### Configuration JSON Format

Each configuration document should follow this structure (see `mongo_mcp_annotations.json` for complete examples):

```json
{
    "Name": "AirbnbSearch",
    "module_info": {
        "title": "MongoDB Vector Search MCP Server",
        "description": "A fastMCP MCP server that provides vector search capabilities using MongoDB's $search aggregation pipeline.",
        "database": "sample_airbnb",
        "collection": "listingsAndReviews"
    },
    "tools": {
        "vector_search": {      
            "description": "Perform semantic vector similarity search on MongoDB collection using AI embeddings.",
            "index": "listing_vector_index",
            "required": ["query_text"],
            "parameters": {
                "query_text": {
                    "type": "str",
                    "description": "Natural language query describing desired property characteristics."
                },
                "limit": {
                    "type": "int",
                    "default": 10,
                    "constraints": "ge=1, le=50",
                    "description": "Maximum number of results to return (default: 10, max recommended: 50)"
                }
            },
            "projection": {
                "embedding": 0,
                "images": 0
            },
            "returns": "JSON with results array containing matching properties ranked by semantic similarity."
        },
        "get_unique_values": {
            "description": "Get unique values for a specific field in the MongoDB collection.",
            "required": ["field"],
            "parameters": {
                "field": {
                    "type": "str",
                    "description": "The field name to get unique values for."
                }
            },
            "returns": "JSON with unique values array for the specified field."
        }
    }
}
```

### Configuration Fields

- **Name**: Unique identifier for the MCP server configuration
- **module_info**: Metadata about the server including:
  - `title`: Display title for the MCP server
  - `description`: Description of the server's purpose
  - `database`: Target MongoDB database name
  - `collection`: Target MongoDB collection name
- **tools**: Object containing tool definitions where each key is the tool name and value contains:
  - `description`: Detailed description of what the tool does
  - `required`: Array of required parameter names
  - `parameters`: Object defining each parameter with type, description, defaults, and constraints
  - `returns`: Description of what the tool returns
  - `index`: (for search tools) MongoDB index name to use
  - `projection`: (optional) MongoDB projection to exclude/include fields

### Example Configurations

The `mongo_mcp_annotations.json` file contains two complete examples:

1. **AirbnbSearch**: Full-featured configuration with vector search, text search, and data analysis tools for Airbnb property data
2. **NetflixSearch**: Simplified configuration with basic data exploration tools for movie data

### Loading Configurations

The MCP server will:
1. Connect to the configuration MongoDB collection
2. Load the specified configuration document by name
3. Dynamically generate MCP tools based on the configuration
4. Connect to the target database/collection specified in the configuration
5. Expose the configured tools via the MCP protocol

This approach allows you to:
- Create multiple MCP servers for different datasets
- Modify tool behavior without code changes
- Add new tools by updating the configuration
- Customize tool parameters and descriptions for specific use cases

## Target Data Configuration

While the MCP server configuration is stored in a dedicated collection, the actual data being searched resides in target collections. Here's an example using the [MongoDB Atlas Sample Airbnb Dataset](https://www.mongodb.com/docs/atlas/sample-data/sample-airbnb/). 

For vector search capabilities, the target collection should have documents with the following structure:

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

1. **Create a virtual environment**:
   ```bash
   python -m venv .
   ```

2. **Activate the virtual environment**:
   ```bash
   source bin/activate
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
   ```

5. **Add a AWS Secrets Manager Key**
   The `MONGO_CREDS` secret name should match the MONGO_CREDS env variable. The value should contain:
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
           "MONGO_CREDS": "your-secrets-manager-secret-name"           
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
           "MONGO_CREDS": "your-secrets-manager-secret-name"
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
