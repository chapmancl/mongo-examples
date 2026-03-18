#!/bin/bash
VERSION="1"
AWS_REGION="us-east-2"
AWS_ACCOUNT_ID=""
AWS_ECS_CLUSTER=""
AWS_ECS_SERVICE="mcpwebui-service"
AWS_EKS_CLUSTER="mcp-mongo-cluster"
AWS_EKS_NAMESPACE="mcp-mongo-app"
ECR_MCP_REPOSITORY="mongodb-mcp-service"
ECR_WEBUI_REPOSITORY="mongodb-mcp-webui"
AWS_ECR_PREFIX=""

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# build the MCP server image and push to ECR
docker build -t $ECR_MCP_REPOSITORY .
#docker tag $ECR_MCP_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_MCP_REPOSITORY:v$VERSION
#docker tag $ECR_MCP_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_MCP_REPOSITORY:latest
#docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_MCP_REPOSITORY:latest
#docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_MCP_REPOSITORY:v$VERSION

# build the webui image and push to ECR
VERSION="1"
cd ../webui
docker build -t $ECR_WEBUI_REPOSITORY . --build-context rootfolder=../
#docker tag $ECR_WEBUI_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_WEBUI_REPOSITORY:v$VERSION
#docker tag $ECR_WEBUI_REPOSITORY:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_WEBUI_REPOSITORY:latest
#docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_WEBUI_REPOSITORY:latest
#docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$AWS_ECR_PREFIX/$ECR_WEBUI_REPOSITORY:v$VERSION

# force update with new images
#aws ecs update-service --cluster $AWS_ECS_CLUSTER --service $AWS_ECS_SERVICE --force-new-deployment --region $AWS_REGION
#kubectl rollout restart deployment -n $AWS_EKS_NAMESPACE -c $AWS_EKS_CLUSTER