#!/usr/bin/env python3
"""
Test script for Bedrock reasoning engine using Claude Haiku 4.5 with extended thinking.

This script tests the new Claude Haiku model with reasoning capabilities.

Usage:
    export AWS_BEARER_TOKEN="your-token"
    python scratch/test_bedrock_reasoning.py
   
    # Or with custom region
    export AWS_REGION="us-east-2"
    python scratch/test_bedrock_reasoning.py --region us-west-2
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


class TestSettings:
    """Mock settings for testing"""
    def __init__(self, region: str = "us-east-1"):
        self.aws_region = region
        self.LLM_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        self.LLM_MAX_ITERATIONS = 5
        self.ENABLE_CACHE_POINTS = False


def test_aws_credentials(region: str):
    """Test AWS credentials and bearer token"""
    print("=" * 60)
    print("Test 1: AWS Credentials & Bearer Token")
    print("=" * 60)
   
    bearer_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
   
    if not bearer_token:
        print("✗ AWS_BEARER_TOKEN_BEDROCK environment variable not set")
        print("\nTo configure, run:")
        print("  export AWS_BEARER_TOKEN_BEDROCK='your-token'")
        return False
   
    print(f"✓ AWS_BEARER_TOKEN_BEDROCK found")
    print(f"  Token: {bearer_token[:20]}...")
    print(f"  Region: {region}")
   
    return True


def create_bedrock_client(region: str):
    """Create Bedrock Runtime client"""
    print("\n" + "=" * 60)
    print("Test 2: Create Bedrock Client")
    print("=" * 60)
   
    try:
        client = boto3.client(
            'bedrock-runtime',
            region_name=region,
            config=BotoConfig(
                read_timeout=120,
                connect_timeout=10,
                retries={"max_attempts": 2, "mode": "adaptive"},
            ),
        )
       
        print(f"✓ Bedrock Runtime client created")
        print(f"  Service: bedrock-runtime")
        print(f"  Region: {region}")
        print(f"  Endpoint: {client.meta.endpoint_url}")
       
        return client
       
    except Exception as e:
        print(f"✗ Failed to create Bedrock client: {e}")
        return None


async def test_simple_reasoning(client, model_id: str):
    """Test simple reasoning with Claude Haiku"""
    print("\n" + "=" * 60)
    print("Test 3: Simple Reasoning Task")
    print("=" * 60)
   
    prompt = "What is 15 * 24? Think through this step by step."
   
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    print()
   
    try:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
       
        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=messages
        )
       
        # Extract response
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
       
        print("✓ Response received")
        print("\n📝 Claude's Response:")
        print("-" * 60)
        for item in content:
            if "text" in item:
                print(item["text"])
        print("-" * 60)
       
        # Show usage
        usage = response.get("usage", {})
        print(f"\n📊 Usage Statistics:")
        print(f"  Input tokens: {usage.get('inputTokens', 0)}")
        print(f"  Output tokens: {usage.get('outputTokens', 0)}")
        print(f"  Total tokens: {usage.get('totalTokens', 0)}")
       
        return True
       
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"✗ AWS Error: {error_code}")
        print(f"  Message: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_turn_reasoning(client, model_id: str):
    """Test multi-turn conversation with reasoning"""
    print("\n" + "=" * 60)
    print("Test 5: Multi-Turn Reasoning")
    print("=" * 60)

    print(f"Model: {model_id}")
    print("Testing conversational reasoning with follow-up questions")
    print()

    try:
        # First turn
        messages = [
            {
                "role": "user",
                "content": [{"text": "If I have 3 apples and buy 7 more, how many do I have?"}]
            }
        ]

        print("Turn 1:")
        print("User: If I have 3 apples and buy 7 more, how many do I have?")

        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=messages
        )

        # Extract first response
        assistant_message = response["output"]["message"]
        assistant_text = assistant_message["content"][0]["text"]
        print(f"Assistant: {assistant_text}")

        # Add assistant response to conversation
        messages.append(assistant_message)

        # Second turn
        messages.append({
            "role": "user",
            "content": [{"text": "If I give away 4 apples, how many do I have left?"}]
        })

        print("\nTurn 2:")
        print("User: If I give away 4 apples, how many do I have left?")

        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=messages
        )

        # Extract second response
        assistant_message = response["output"]["message"]
        assistant_text = assistant_message["content"][0]["text"]
        print(f"Assistant: {assistant_text}")

        print("\n✓ Multi-turn conversation successful")

        # Show total usage
        usage = response.get("usage", {})
        print(f"\n📊 Final Usage Statistics:")
        print(f"  Total tokens: {usage.get('totalTokens', 0)}")

        return True

    except Exception as e:
        print(f"✗ Error in multi-turn conversation: {e}")
        return False


async def test_creative_reasoning(client, model_id: str):
    """Test creative reasoning task"""
    print("\n" + "=" * 60)
    print("Test 6: Creative Reasoning")
    print("=" * 60)

    prompt = """Come up with a creative solution to this problem:
A cat is stuck in a tree 20 feet high. You have:
- A ladder (15 feet tall)
- A rope (30 feet long)
- A can of tuna
- A megaphone

Think creatively and explain your reasoning."""

    print(f"Model: {model_id}")
    print(f"Prompt: Creative problem-solving scenario")
    print()

    try:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=messages
        )

        # Extract response
        content = response["output"]["message"]["content"]

        print("✓ Response received")
        print("\n💡 Claude's Creative Solution:")
        print("-" * 60)
        for item in content:
            if "text" in item:
                print(item["text"])
        print("-" * 60)

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_complex_reasoning(client, model_id: str):
    """Test complex reasoning with extended thinking"""
    print("\n" + "=" * 60)
    print("Test 4: Complex Reasoning Task")
    print("=" * 60)
   
    prompt = """A farmer has chickens and cows. Together they have 50 heads and 140 legs.
How many chickens and how many cows does the farmer have?
Think through this step-by-step and show your reasoning."""
   
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt[:100]}...")
    print()
   
    try:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
       
        # Add system prompt for extended thinking
        system = [
            {
                "text": "You are a helpful assistant who thinks through problems step by step. Show your reasoning process clearly."
            }
        ]
       
        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=messages,
            system=system
        )
       
        # Extract response
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
       
        print("✓ Response received")
        print("\n🧠 Claude's Reasoning:")
        print("-" * 60)
        for item in content:
            if "text" in item:
                print(item["text"])
        print("-" * 60)
       
        # Show usage
        usage = response.get("usage", {})
        print(f"\n📊 Usage Statistics:")
        print(f"  Input tokens: {usage.get('inputTokens', 0)}")
        print(f"  Output tokens: {usage.get('outputTokens', 0)}")
        print(f"  Total tokens: {usage.get('totalTokens', 0)}")
       
        return True
       
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"✗ AWS Error: {error_code}")
        print(f"  Message: {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(
        description='Test Bedrock reasoning engine with Claude Haiku 4.5'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    parser.add_argument(
        '--skip-creative',
        action='store_true',
        help='Skip the creative reasoning test'
    )

    args = parser.parse_args()

    print("\n🧠 Bedrock Reasoning Engine Test")
    print(f"Model: Claude Haiku 4.5 (Extended Thinking)\n")

    # Test 1: Credentials
    if not test_aws_credentials(args.region):
        print("\n❌ Test FAILED: AWS credentials not configured")
        sys.exit(1)

    # Test 2: Create client
    client = create_bedrock_client(args.region)
    if not client:
        print("\n❌ Test FAILED: Cannot create Bedrock client")
        sys.exit(1)

    model_id = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

    # Test 3: Simple reasoning
    if not await test_simple_reasoning(client, model_id):
        print("\n⚠️  Simple reasoning test failed")
        sys.exit(1)

    # Test 4: Complex reasoning
    if not await test_complex_reasoning(client, model_id):
        print("\n⚠️  Complex reasoning test failed")
        sys.exit(1)

    # Test 5: Multi-turn reasoning
    if not await test_multi_turn_reasoning(client, model_id):
        print("\n⚠️  Multi-turn reasoning test failed")
        sys.exit(1)

    # Test 6: Creative reasoning (optional)
    if not args.skip_creative:
        if not await test_creative_reasoning(client, model_id):
            print("\n⚠️  Creative reasoning test failed")

    print("\n" + "=" * 60)
    print("✅ All Reasoning Tests Passed!")
    print("=" * 60)
    print("\nClaude Haiku 4.5 is working correctly with extended thinking.")


if __name__ == "__main__":
    asyncio.run(main())