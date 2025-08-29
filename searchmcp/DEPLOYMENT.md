# MongoDB Vector Search MCP Server - ECS Fargate Deployment Guide

This guide walks you through deploying the MongoDB Vector Search MCP Server as an ECS Fargate service that can be used with Amazon Bedrock Agents.

## Architecture Overview

The deployment creates:
- **VPC** with public and private subnets across 2 AZs
- **ECS Fargate Cluster** running the MCP server
- **ECR Repository** for Docker images
- **Application Load Balancer** for external access (optional)
- **CloudWatch Logs** for monitoring
- **IAM Roles** with least privilege access

## Prerequisites

1. **AWS CLI** installed and configured
   ```bash
   aws configure
   ```

2. **Docker** installed and running

3. **Terraform** >= 1.0 installed

4. **MongoDB Atlas** cluster with vector search index configured

5. **AWS Permissions**: Your AWS credentials need permissions for:
   - ECS (create clusters, services, task definitions)
   - ECR (create repositories, push images)
   - VPC (create VPCs, subnets, security groups)
   - IAM (create roles and policies)
   - CloudWatch (create log groups)
   - Application Load Balancer

## Quick Start

### 1. Configure Variables

Copy the example variables file and update with your values:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
# AWS Configuration
aws_region = "us-east-1"

# Project Configuration
project_name = "mongodb-vector-mcp"
environment  = "prod"

# MongoDB Configuration - UPDATE THESE VALUES
mongo_uri         = "mongodb+srv://your-username:your-password@your-cluster.mongodb.net/"
mongo_database    = "demo1"
vector_collection = "sample_airbnb"
vector_index      = "listing_vector_index"
search_index      = "default"

# ECS Configuration
cpu           = 256
memory        = 512
desired_count = 1
```

### 2. Deploy (Linux/macOS)

```bash
# Make deployment script executable
chmod +x deploy.sh

# Full deployment
./deploy.sh deploy

# Or step by step:
./deploy.sh build      # Build and push Docker image only
./deploy.sh update     # Update existing service with new image
./deploy.sh outputs    # Show deployment outputs
./deploy.sh destroy    # Clean up all resources
```

### 3. Deploy (Windows)

```powershell
# Build and push Docker image
$PROJECT_NAME = "mongodb-vector-mcp"
$AWS_REGION = "us-east-1"
$AWS_ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$ECR_REPOSITORY_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$PROJECT_NAME"

# Create ECR repository
aws ecr create-repository --repository-name $PROJECT_NAME --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

# Build and push image
docker build -t ${PROJECT_NAME}:latest .
docker tag ${PROJECT_NAME}:latest ${ECR_REPOSITORY_URI}:latest
docker push ${ECR_REPOSITORY_URI}:latest

# Deploy infrastructure
terraform init
terraform plan
terraform apply
```

## Manual Deployment Steps

### 1. Build and Push Docker Image

```bash
# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="us-east-1"
PROJECT_NAME="mongodb-vector-mcp"
ECR_REPOSITORY_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"

# Create ECR repository
aws ecr create-repository --repository-name ${PROJECT_NAME} --region ${AWS_REGION}

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY_URI}

# Build Docker image
docker build -t ${PROJECT_NAME}:latest .

# Tag and push image
docker tag ${PROJECT_NAME}:latest ${ECR_REPOSITORY_URI}:latest
docker push ${ECR_REPOSITORY_URI}:latest
```

### 2. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Review deployment plan
terraform plan

# Deploy infrastructure
terraform apply
```

### 3. Verify Deployment

```bash
# Check ECS service status
aws ecs describe-services \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --services $(terraform output -raw ecs_service_name)

# Check task logs
aws logs tail /ecs/mongodb-vector-mcp --follow
```

## Integration with Amazon Bedrock Agents

Once deployed, you can integrate the MCP server with Amazon Bedrock Agents:

### Option 1: Using ECS Service Discovery (Recommended)

```python
from mcp.client.stdio import MCPStdio, StdioServerParameters
from inline_agent import InlineAgent, ActionGroup

# Configure MCP server parameters for ECS deployment
mongodb_server_params = StdioServerParameters(
    command="aws",
    args=[
        "ecs", "execute-command",
        "--cluster", "mongodb-vector-mcp-cluster",
        "--task", "task-id",  # Get from ECS console or CLI
        "--container", "mongodb-vector-mcp",
        "--interactive",
        "--command", "python mcp.py"
    ],
    env={}
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
```

### Option 2: Using Load Balancer Endpoint

If you need HTTP access, the deployment includes an Application Load Balancer:

```python
# Get the load balancer DNS name
load_balancer_dns = terraform output -raw load_balancer_dns_name

# Use HTTP client to communicate with MCP server
# (You'll need to modify the MCP server to support HTTP if needed)
```

## Monitoring and Troubleshooting

### View Logs

```bash
# View ECS service logs
aws logs tail /ecs/mongodb-vector-mcp --follow

# View specific log stream
aws logs describe-log-streams --log-group-name /ecs/mongodb-vector-mcp
```

### Check Service Health

```bash
# Check ECS service status
aws ecs describe-services \
  --cluster mongodb-vector-mcp-cluster \
  --services mongodb-vector-mcp

# Check running tasks
aws ecs list-tasks \
  --cluster mongodb-vector-mcp-cluster \
  --service-name mongodb-vector-mcp
```

### Common Issues

1. **Task fails to start**: Check CloudWatch logs for error messages
2. **MongoDB connection issues**: Verify MongoDB URI and network connectivity
3. **Image pull errors**: Ensure ECR repository exists and image was pushed
4. **Permission errors**: Check IAM roles have necessary permissions

## Scaling

### Horizontal Scaling

```bash
# Update desired count
aws ecs update-service \
  --cluster mongodb-vector-mcp-cluster \
  --service mongodb-vector-mcp \
  --desired-count 3
```

### Vertical Scaling

Update the Terraform variables:

```hcl
cpu    = 512   # Increase CPU
memory = 1024  # Increase memory
```

Then apply changes:

```bash
terraform apply
```

## Security Considerations

1. **Network Security**: The ECS tasks run in private subnets with no direct internet access
2. **IAM Roles**: Uses least privilege access with specific permissions
3. **Secrets Management**: Consider using AWS Secrets Manager for MongoDB credentials
4. **VPC Security**: Security groups restrict access to necessary ports only

## Cost Optimization

1. **Fargate Spot**: Consider using Fargate Spot for non-production workloads
2. **Right-sizing**: Monitor CPU/memory usage and adjust task definition
3. **Auto Scaling**: Implement ECS auto scaling based on metrics
4. **Reserved Capacity**: Use Savings Plans for predictable workloads

## Cleanup

To remove all resources:

```bash
# Using deployment script (Linux/macOS)
./deploy.sh destroy

# Using Terraform directly
terraform destroy
```

## Support

For issues and questions:
1. Check CloudWatch logs for error messages
2. Verify MongoDB connectivity and index configuration
3. Ensure all prerequisites are met
4. Review AWS service limits and quotas

## Next Steps

After successful deployment:
1. Test the MCP server tools using Amazon Bedrock Agents
2. Set up monitoring and alerting
3. Configure auto-scaling policies
4. Implement CI/CD pipeline for updates
5. Set up backup and disaster recovery procedures
