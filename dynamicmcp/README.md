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
4. Run fastapi (see [FastAPI Deployment](#fastapi-deployment))

## Dynamic Configuration Setup

This MCP server dynamically loads its configuration from a MongoDB collection. The configuration defines which tools are available, their parameters, descriptions, and behavior. This allows you to create multiple MCP server configurations for different databases and collections without modifying code.

### Configuration Collection

Create a MongoDB collection to store your MCP configurations (e.g., `mcp_tools`). Each document in this collection defines a complete MCP server configuration.

### Configuration JSON Format

Each configuration document should follow this structure (see `mcp_config.mcp_tools.json` for complete examples):

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

The `mcp_config.mcp_tools.json` file contains two complete examples:

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
   export AWS_REGION=your-aws-region
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

## FastAPI Deployment

This server can be deployed using [FastAPI](https://gofastapi.com/) for enhanced deployment options and multiple transport capabilities.
For more FastAPI deployment options, see the [FastAPI documentation](https://gofastapi.com/deployment/running-server).

**Default HTTP Transport (for production/containers):**
```bash
fastapi run mongo_mcp.py
```

The server will start with HTTP transport on port 8000 at `http://localhost:8000/mcp/`.

### Docker Instructions

Build the Docker image for the MCP server:

```bash
docker build -t mongodb-vector-mcp .
```

### Pushing to Amazon ECR

1. **Create an ECR repository** (if it doesn't exist):
   ```bash
   aws ecr create-repository --repository-name mongodb-vector-mcp --region your-aws-region
   ```

2. **Get the login token and authenticate Docker to ECR**:
   ```bash
   aws ecr get-login-password --region your-aws-region | docker login --username AWS --password-stdin <account-id>.dkr.ecr.your-aws-region.amazonaws.com
   ```

3. **Tag the image for ECR**:
   ```bash
   docker tag mongodb-vector-mcp:latest <account-id>.dkr.ecr.your-aws-region.amazonaws.com/mongodb-vector-mcp:latest
   ```

4. **Push the image to ECR**:
   ```bash
   docker push <account-id>.dkr.ecr.your-aws-region.amazonaws.com/mongodb-vector-mcp:latest
   ```

Replace `<account-id>` with your AWS account ID.

You can run the server in Docker with FastMCP HTTP transport by with the Dockerfile CMD:

```dockerfile
CMD ["fastapi", "run", "mongo_mcp.py"]
```

Then run the container locally with port mapping:

```bash
docker run -p 8000:8000 \
  -e AWS_REGION=your-aws-region \
  -e MONGO_CREDS=your-secret-name \
  mongodb-vector-mcp:latest
```

## Kubernetes Deployment with Terraform

This project includes Terraform configuration for deploying the MCP server to Amazon EKS (Elastic Kubernetes Service). The Terraform deployment creates multiple MCP services based on different MongoDB configurations.

### Prerequisites for Kubernetes Deployment

- AWS CLI configured with appropriate permissions
- Terraform installed (version 1.0+)
- An existing EKS cluster
- AWS Load Balancer Controller installed in your EKS cluster
- IAM role for Service Account (IRSA) configured for accessing AWS Secrets Manager
- SSL certificate in AWS Certificate Manager (optional, for HTTPS)

### Terraform Configuration

The `main.tf` file provides a complete infrastructure setup including:

- **Multiple Service Deployment**: Deploy multiple MCP server instances with different configurations
- **Load Balancing**: AWS Application Load Balancer with SSL termination
- **Service Discovery**: Kubernetes services and ingress rules for routing
- **AWS Integration**: IAM roles for Service Account (IRSA) for Secrets Manager access
- **Health Checks**: Configured health check endpoints for each service

### Configuration Variables

Create a `terraform.tfvars` file based on `terraform.tfvars.example`:

```hcl
# List of MCP configurations to deploy
services = ["AirbnbSearch", "WeatherSearch", "MflixSearch"]

# Kubernetes configuration
namespace = "mcp-search-app"
cluster_name = "your-eks-cluster-name"
K8service_account = "mcp-mcp-sa"

# AWS configuration
aws_region = "your-aws-region"
ecr_repository = "your-account-id.dkr.ecr.your-aws-region.amazonaws.com/mongodb-dynamic-mcp"
image_tag = "latest"

# IAM and SSL
iam_role_arn = "arn:aws:iam::your-account-id:role/YourMCPRole"
certificate_arn = "arn:aws:acm:your-aws-region:your-account-id:certificate/your-cert-arn"

# MongoDB credentials
mongo_creds = "your-secrets-manager-secret-name"
```

### Deploying to Kubernetes

1. **Build and push the Docker image** to ECR:
   ```bash
   # Build the image
   docker build -t mongodb-dynamic-mcp .
   
   # Tag for ECR
   docker tag mongodb-dynamic-mcp:latest your-account-id.dkr.ecr.your-aws-region.amazonaws.com/mongodb-dynamic-mcp:latest
   
   # Push to ECR
   docker push your-account-id.dkr.ecr.your-aws-region.amazonaws.com/mongodb-dynamic-mcp:latest
   ```

2. **Initialize and apply Terraform**:
   ```bash
   # Initialize Terraform
   terraform init
   
   # Review the deployment plan
   terraform plan
   
   # Apply the configuration
   terraform apply
   ```

3. **Access your deployed services**:
   - Each MCP configuration will be available at `https://your-domain/ServiceName`
   - For example: `https://your-alb-url/AirbnbSearch`, `https://your-alb-url/WeatherSearch`
   - The default catch-all route distributes traffic across all services

### Environment Variables in Kubernetes

Each deployed service automatically receives:

- `MCP_TOOL_NAME`: The specific configuration name (e.g., "AirbnbSearch")
- `MONGO_CREDS`: Reference to AWS Secrets Manager secret
- `AWS_REGION`: AWS region for Secrets Manager access

The service will dynamically load the appropriate configuration from MongoDB based on the `MCP_TOOL_NAME` environment variable.

## Example MongoDB Configuration Import

This repository includes a ready-to-use MongoDB configuration file (`mcp_config.mcp_tools.json`) that contains example configurations for different datasets. This file can be imported directly into your MongoDB configuration collection to get started quickly.

### Configuration Examples Included

The `mcp_config.mcp_tools.json` file contains three example configurations:

1. **AirbnbSearch** - Full-featured configuration for searching Airbnb property listings
   - Vector search capabilities for semantic similarity
   - Text search for keyword-based queries  
   - Unique value discovery for filters
   - Custom aggregation queries
   - Collection metadata access

2. **MflixSearch** - Configuration for Netflix movie data analysis
   - Data exploration tools for movie collections
   - Aggregation pipeline support
   - Field analysis capabilities

3. **WeatherSearch** - Configuration for historical weather data
   - Weather data analysis tools
   - Time-series data exploration
   - Custom aggregation support

### Importing the Configuration

To import the example configurations into your MongoDB collection:

1. **Using MongoDB Compass**:
   - Open MongoDB Compass and connect to your cluster
   - Navigate to your configuration database and collection
   - Click "Add Data" → "Import JSON or CSV file" 
   - Select the `mcp_config.mcp_tools.json` file
   - Choose "JSON" as the file type and import

2. **Using MongoDB CLI (mongoimport)**:
   ```bash
   mongoimport --uri "mongodb+srv://username:password@cluster.mongodb.net/your_config_db" \
     --collection mcp_configurations \
     --file mcp_config.mcp_tools.json \
     --jsonArray
   ```

3. **Using MongoDB Shell**:
   ```javascript
   // Connect to your MongoDB cluster
   use your_config_database
   
   // Load and insert the configuration data
   const configs = [/* paste content from mcp_config.mcp_tools.json */];
   db.mcp_configurations.insertMany(configs);
   ```

### Customizing the Examples

After importing, you can customize these configurations:

- **Database/Collection Names**: Update the `module_info.database` and `module_info.collection` fields to point to your data
- **Tool Parameters**: Modify tool parameters to match your specific use case
- **Index Names**: Update vector and text search index names to match your MongoDB indexes
- **Field Names**: Adjust projection fields and filter options based on your data schema
- **Active Status**: Set `active: true/false` to enable/disable specific configurations

### Using Different Configurations

Each imported configuration can be used by setting the `MCP_TOOL_NAME` environment variable:

```bash
# Use the AirbnbSearch configuration
export MCP_TOOL_NAME=AirbnbSearch
fastapi run mongo_mcp.py

# Use the MflixSearch configuration  
export MCP_TOOL_NAME=MflixSearch
fastapi run mongo_mcp.py

# Use the WeatherSearch configuration
export MCP_TOOL_NAME=WeatherSearch
fastapi run mongo_mcp.py
```

This allows you to run multiple MCP server instances, each configured for different datasets and use cases, all from the same codebase.

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
           "AWS_REGION": "your-aws-region",
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
      "command": "fastapi",
      "args": ["run", "/path/to/your/mongo_mcp.py"],
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
         "command": "fastapi",
         "args": ["run", "mongo_mcp.py", "--transport", "sse", "--port", "8001"],
         "cwd": "/path/to/your/mongodb-mcp-project",
         "env": {
           "AWS_REGION": "your-aws-region",
           "MONGO_CREDS": "your-secrets-manager-secret-name"
         }
       }
     }
   }
   ```

3. **Alternative Configuration** (if running the server separately):
   If you prefer to run the server manually, start it with:
   ```bash
   fastapi run mongo_mcp.py --transport sse --port 8001
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
