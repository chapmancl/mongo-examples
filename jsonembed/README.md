# MongoDB Vector Search with RAG Example

This repository demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** system using MongoDB Atlas Vector Search and AWS Bedrock. The example uses the Airbnb sample dataset to showcase:

- **`embedairbnb.py`** - Generates vector embeddings for documents and stores them in MongoDB
- **`searchairbnb.py`** - Provides an interactive RAG system with vector search and LLM integration

For more advanced AI conversations that can build queries and self-discover data interactions, see the MCP server examples in the `mcpclient/` directory.

## 1. Setup Python Environment

```bash
python3 -m venv .
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Set Your MongoDB URI

Edit the `settings.py` file:

```python
# settings.py
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster-url>/test"
```

Replace `<username>`, `<password>`, and `<cluster-url>` with your credentials.

## 3. Create a Vector Index

In your MongoDB collection, create a vector index (e.g., using the Atlas UI or with PyMongo):
To create a vector index in MongoDB Atlas, use the following JSON configuration in the Atlas UI. This example includes filter options as used in `searchairbnb.py`
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

## 4. Generate Embeddings

Use the `embedairbnb.py` script to generate embeddings for your documents and store them in MongoDB:
```bash
# Activate your virtual environment
source venv/bin/activate

# Run the embedding script
python embedairbnb.py
```

### What the Script Does
1. **Document Processing**: Retrieves documents from the source collection
2. **Field Selection**: Extracts key fields (name, summary, description, property_type, room_type, bed_type, address)
3. **Text Chunking**: Converts selected fields into a single text chunk per document
4. **Vectorization**: Generates embeddings using AWS Bedrock's Titan model
5. **Storage**: Updates documents with embedding vectors in the target collection

### Customization Options
- **Document Limit**: By default, processes 6000 documents. Modify the `main()` function to change this:
  ```python
  vectorizer.process_documents(1000)  # Process 1000 documents
  ```
- **Field Selection**: Edit the `process_documents()` method to include/exclude fields
- **Chunking Strategy**: Use `extract_fields()` method for more granular chunking instead of `chunk_entire_doc()`

### Monitoring Progress
The script outputs processing time for each document and total completion time:
```
507f1f77bcf86cd799439011 completed in 0.8234 seconds.
507f1f77bcf86cd799439012 completed in 0.7891 seconds.
...
6000 completed in 4523.1234 seconds.
```

## 5. Run the Search and Query System

Use the `searchairbnb.py` script to perform vector searches and interact with the LLM using your embedded Airbnb dataset:

```bash
# Activate your virtual environment
source venv/bin/activate

# Run the search script
python searchairbnb.py
```

### What the Script Does
The `searchairbnb.py` script provides a **Retrieval-Augmented Generation (RAG)** system that:

1. **Vector Search**: Converts your questions into embeddings and finds similar Airbnb listings
2. **Metadata Filtering**: Applies filters based on your query (country, market, beds, bedrooms, listing ID)
3. **LLM Integration**: Uses AWS Bedrock's Claude model to generate natural language responses
4. **Conversation History**: Maintains context across multiple questions in a session


### Query Examples

#### Basic Questions
```
Question: What are some beachfront properties?
Question: Show me listings with great reviews
Question: Find properties good for families
```

#### Filtered Searches
Use these filter patterns in your questions:
```
Question: country=US show me properties in New York
Question: market=Paris beds=2 find apartments for couples
Question: bedrooms=3 country=AU what's available in Sydney?
Question: id=12345678 show me details for this specific listing
```

#### Direct Claude Queries
```
Question: ask what makes a good Airbnb host?
Question: ask explain the difference between entire home and private room
```

### Filter Options
The system automatically detects and applies these filters from your questions:

- **`country=XX`** - Filter by country code (e.g., US, FR, AU)
- **`market=CityName`** - Filter by market/city
- **`beds=N`** - Filter by number of beds
- **`bedrooms=N`** - Filter by number of bedrooms  
- **`id=XXXXXXXX`** - Get specific listing by ID (skips vector search)

### Sample Session
```
Enter questions (Press Ctrl+C to stop):
Commands:
  ask <question> - Direct Claude query without vector search
  <question> - Full query with vector search and Claude (classic RAG)
  clear - Clear conversation history

Question: country=US beds=2 show me properties in beach locations
Answer: Based on the search results, here are some great 2-bed beachfront properties in the US...

Question: What about the pricing for these properties?
Answer: Looking at the properties from your previous search, the pricing varies...

Question: ask what should I look for when booking an Airbnb?
Answer: When booking an Airbnb, here are the key factors to consider...

Question: clear
Answer: history cleared...
```

### Performance Features

- **Timing Information**: The script displays processing times for each operation
- **Smart Filtering**: Automatically optimizes search based on detected filters
- **History Management**: Automatically trims conversation history to prevent token overflow
- **Error Handling**: Gracefully handles AWS token expiration and validation errors

### Customization Options

You can modify the search behavior by editing these parameters in the `retrieve_aggregate_facts()` method:

```python
limit = 5      # Maximum results to return (default: 5)
candidates = 400  # Number of candidates to evaluate (default: 400)
```

### Troubleshooting

- **AWS Token Expired**: The script will notify you and exit. Refresh your AWS credentials.
- **Too Much History**: The system automatically clears history when token limits are reached.
- **No Results**: Try broader search terms or remove filters.
- **Connection Issues**: Verify your MongoDB URI and AWS credentials in `settings.py`.

### Exit the Program
Press `Ctrl+C` to stop the interactive session.
