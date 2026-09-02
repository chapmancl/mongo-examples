#!/usr/bin/env python3
"""
Quick test script to verify boto3 connection to AWS Bedrock.
Tests both basic connectivity and embedding generation.
"""

import os 
import sys 
import json 
import boto3 
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError 
import voyageai 
from pymongo import MongoClient 
from pymongo.errors import ConnectionFailure, OperationFailure 
# apppend parent folder to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(base_dir))
sys.path.append(os.path.dirname(os.path.dirname(base_dir)))


if os.environ.get("USE_LOCAL_MODE") == "true":
    from lmig_settings import settings
else:
    from lmig_settings import settings 

def test_aws_credentials():
    """Test if AWS credentials are configured"""
    print("=" * 60)
    print("Test 1: AWS Credentials")
    print("=" * 60)
    try:
        # Try to get credentials 
        session = boto3.Session()
        credentials = None #session.get_credentials() 
        if credentials is None:
            print("❌ No AWS credentials found")
            print("\nTo configure credentials, run:")
            print(" aws configure")
            print("\nOr set environment variables:")
            print(" export AWS_ACCESS_KEY_ID=your_key")
            print(" export AWS_SECRET_ACCESS_KEY=your_secret")
            return False 
        print(f"✓ AWS credentials found")
        print(f" Access Key ID: {credentials.access_key[:10]}...")
        # Get region 
        region = session.region_name or os.getenv('AWS_REGION', 'us-east-2')
        print(f" Region: {region}")
        return True 
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")
        return False 


def test_voyage_embedding():
    """Test if Voyage embedding is working"""
    # Initialize Voyage AI client 
    if settings.voyage_api_key:
        voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    else:
        # Will use VOYAGE_API_KEY environment variable 
        voyage_client = voyageai.Client()
    result = get_embedding(voyage_client, "This is a test sentence for embedding generation.", "query")
    print(f"Embedding generated successfully: {result[:5]}...")

def get_embedding(v_client, text: str, input_type: str = "query") -> list[float]:
    """
    Generate embedding for a given text using Voyage AI.
    Args:
    text: Text to embed
    input_type: Type of input ("query" or "document")
    Returns:
    List[float]: Embedding vector
    """
    try:
        result = v_client.embed(
            texts=[text],
            model=settings.voyage_model,
            input_type=input_type 
        )
        return result.embeddings[0]
    except Exception as e:
        print(f"Error generating embedding: {str(e)}", "ERROR")
        raise 

def test_bedrock_runtime_client():
    """Test creating Bedrock Runtime client"""
    print("\n" + "=" * 60)
    print("Test 2: Bedrock Runtime Client")
    print("=" * 60)
    try:
        region = os.getenv('AWS_REGION', 'us-east-2')
        client = boto3.client('bedrock-runtime', region_name=region)
        print(f"✓ Bedrock Runtime client created successfully")
        print(f" Region: {region}")
        print(f" Endpoint: {client.meta.endpoint_url}")
        return client
    except NoCredentialsError:
        print("❌ No AWS credentials found")
        return None 
    except PartialCredentialsError:
        print("❌ Incomplete AWS credentials")
        return None 
    except Exception as e:
        print(f"❌ Error creating Bedrock client: {e}")
        return None 


def test_list_foundation_models():
    """Test listing available foundation models"""
    print("\n" + "=" * 60)
    print("Test 3: List Foundation Models")
    print("=" * 60)
    try:
        region = os.getenv('AWS_REGION', 'us-east-2')
        client = boto3.client('bedrock', region_name=region)
        response = client.list_foundation_models()
        models = response.get('modelSummaries', [])
        print(f"✓ Found {len(models)} foundation models")
        # Show embedding models 
        embedding_models = [m for m in models if 'embed' in m.get('modelId', '').lower()]
        print(f"\nEmbedding models available:")
        for model in embedding_models[:5]:
            print(f" - {model['modelId']}")
        # Show Claude models 
        claude_models = [m for m in models if 'claude' in m.get('modelId', '').lower()]
        print(f"\nClaude models available:")
        for model in claude_models[:5]:
            print(f" - {model['modelId']}")
        return True 
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print("❌ Access denied - check IAM permissions for Bedrock")
        else:
            print(f"❌ AWS Error: {error_code} - {e.response['Error']['Message']}")
            return False 
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False 


def test_generate_embedding():
    """Test generating an embedding with Titan"""
    print("\n" + "=" * 60)
    print("Test 4: Generate Embedding")
    print("=" * 60)
    try:
        region = os.getenv('AWS_REGION', 'us-east-2')
        client = boto3.client('bedrock-runtime', region_name=region)
        model_id = "amazon.titan-embed-text-v2:0"
        test_text = "This is a test sentence for embedding generation."
        print(f"Model: {model_id}")
        print(f"Input: {test_text}")
        # Prepare request 
        request_body = json.dumps({
            "inputText": test_text
        })
        # Invoke model 
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            contentType='application/json',
            accept='application/json'
        )
        # Parse response 
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        print(f"✓ Embedding generated successfully")
        print(f" Dimensions: {len(embedding)}")
        print(f" First 5 values: {embedding[:5]}")
        return True 
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDeniedException':
            print("❌ Access denied - check IAM permissions for Bedrock")
        elif error_code == 'ResourceNotFoundException':
            print(f"❌ Model not found: {model_id}")
            print(" Make sure Bedrock model access is enabled in your AWS account")
        else:
            print(f"❌ AWS Error: {error_code} - {e.response['Error']['Message']}")
            return False 
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False 


def main():
    """Run all tests"""
    print("\n🧪 AWS Bedrock Connection Test\n")
    # Test 1: Credentials 
    #if not test_aws_credentials():
    #    print("\n⚠️ Cannot proceed without AWS credentials")
    #    sys.exit(1)
    # Test 2: Client creation 
    client = test_bedrock_runtime_client()
    if not client:
        print("\n⚠️ Cannot proceed without Bedrock client")
        sys.exit(1)
    # Test 3: List models (optional, may fail if no permissions) 
    test_list_foundation_models()
    # Test 4: Generate embedding 
    if test_generate_embedding():
        print("\n" + "=" * 60)
        print("✅ All tests passed! Bedrock is working correctly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ Embedding test failed - check permissions")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main() 
    #test_voyage_embedding()
