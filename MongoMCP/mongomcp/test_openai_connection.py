#!/usr/bin/env python3
"""
Quick test script to verify OpenAI/Azure OpenAI connection.
Tests both basic connectivity and simple text generation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path to import MongoMCP modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from MongoMCP.openai_client import OpenAIClient


class TestSettings:
    """Mock settings class for testing"""
    def __init__(self, use_azure=False):
        if use_azure:
            # Azure OpenAI settings
            self.AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
            self.AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
            self.AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
            self.AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4')
            self.AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-ada-002')
        else:
            # Standard OpenAI settings
            self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            self.OPENAI_MODEL_ID = os.getenv('OPENAI_MODEL_ID', 'gpt-4o-mini')
            self.OPENAI_EMBEDDING_MODEL_ID = os.getenv('OPENAI_EMBEDDING_MODEL_ID', 'text-embedding-3-small')
        
        self.LLM_MAX_ITERATIONS = 10


def test_credentials(use_azure=False):
    """Test if credentials are configured"""
    print("=" * 60)
    print(f"Test 1: {'Azure OpenAI' if use_azure else 'OpenAI'} Credentials")
    print("=" * 60)
    
    if use_azure:
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        if not api_key:
            print("✗ AZURE_OPENAI_API_KEY not found")
            print("\nTo configure Azure OpenAI, set:")
            print("  export AZURE_OPENAI_API_KEY=your_key")
            print("  export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("  export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4")
            return False
        
        if not endpoint:
            print("✗ AZURE_OPENAI_ENDPOINT not found")
            return False
        
        print(f"✓ Azure OpenAI credentials found")
        print(f"  Endpoint: {endpoint}")
        print(f"  API Key: {api_key[:10]}...")
        
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("✗ OPENAI_API_KEY not found")
            print("\nTo configure OpenAI, set:")
            print("  export OPENAI_API_KEY=sk-...")
            print("\nOr for Azure OpenAI, run with --azure flag")
            return False
        
        print(f"✓ OpenAI API key found")
        print(f"  API Key: {api_key[:10]}...")
    
    return True


async def test_client_initialization(use_azure=False):
    """Test creating OpenAI client"""
    print("\n" + "=" * 60)
    print("Test 2: Client Initialization")
    print("=" * 60)
    
    try:
        settings = TestSettings(use_azure=use_azure)
        client = OpenAIClient(settings, use_azure=use_azure)
        
        print(f"✓ OpenAI client created successfully")
        print(f"  Provider: {'Azure OpenAI' if use_azure else 'OpenAI'}")
        print(f"  Model: {client.model_id}")
        print(f"  Embedding Model: {client.embedding_model_id}")
        
        return client
        
    except Exception as e:
        print(f"✗ Error creating OpenAI client: {e}")
        return None


async def test_simple_text(client):
    """Test simple text generation"""
    print("\n" + "=" * 60)
    print("Test 3: Simple Text Generation")
    print("=" * 60)
    
    try:
        prompt = "Say 'Hello, OpenAI is working!'"
        print(f"Prompt: {prompt}")
        
        response = await client.invoke_openai_text(prompt)
        
        print(f"✓ Text generation successful")
        print(f"  Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating text: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tell_joke(client):
    """Test joke generation"""
    print("\n" + "=" * 60)
    print("Test 4: Tell Me a Joke")
    print("=" * 60)
    
    try:
        prompt = "Tell me a short joke about programming."
        print(f"Prompt: {prompt}")
        print()
        
        response = await client.invoke_openai_text(
            prompt=prompt,
            system="You are a funny assistant who tells short, clean jokes."
        )
        
        print(f"✓ Joke generation successful")
        print(f"\n🎭 {response}\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating joke: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_embedding(client):
    """Test embedding generation"""
    print("\n" + "=" * 60)
    print("Test 5: Generate Embedding")
    print("=" * 60)
    
    try:
        test_text = "This is a test sentence for embedding generation."
        print(f"Input: {test_text}")
        
        embedding = await client.generate_embedding(test_text)
        
        print(f"✓ Embedding generated successfully")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating embedding: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test OpenAI/Azure OpenAI connection')
    parser.add_argument('--azure', action='store_true', help='Test Azure OpenAI instead of OpenAI')
    args = parser.parse_args()
    
    print("\n🧪 OpenAI Connection Test\n")
    
    # Test 1: Credentials
    if not test_credentials(use_azure=args.azure):
        print("\n⚠️  Cannot proceed without credentials")
        sys.exit(1)
    
    # Test 2: Client initialization
    client = await test_client_initialization(use_azure=args.azure)
    if not client:
        print("\n⚠️  Cannot proceed without client")
        sys.exit(1)
    
    # Test 3: Simple text
    if not await test_simple_text(client):
        print("\n⚠️  Simple text test failed")
        sys.exit(1)
    
    # Test 4: Tell a joke
    if not await test_tell_joke(client):
        print("\n⚠️  Joke test failed")
        sys.exit(1)
    
    # Test 5: Embedding
    if not await test_embedding(client):
        print("\n⚠️  Embedding test failed")
        sys.exit(1)
    
    print("=" * 60)
    print("✅ All tests passed! OpenAI is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
