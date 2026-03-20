# ============================================================
#  MongoMCP Makefile
#  Usage:  make <target>   (see `make help` for full list)
# ============================================================

# ── Versions ────────────────────────────────────────────────
MCP_VERSION   ?= 1
WEBUI_VERSION ?= 1

# ── ECR / registry ──────────────────────────────────────────
AWS_REGION    ?= us-east-2
AWS_ACCOUNT   ?= 
ECR_REGISTRY  := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
MCP_IMAGE     := mongodb-dynamic-mcp
WEBUI_IMAGE   := mongodb-mcp-ui
MCP_REPO      := $(ECR_REGISTRY)/$(MCP_IMAGE)
WEBUI_REPO    := $(ECR_REGISTRY)/$(WEBUI_IMAGE)

# ── Runtime env (mirrors set-env.sh) ────────────────────────
MCP_TOOL_NAME   ?= airbnbSearch
MONGO_CREDS     ?= 
IS_LOCAL        ?= true
MCP_ROOT_DEV    ?= http://localhost:8000
MCP_ROOT_PROD   ?= https://
WEBUI_EXC       ?= gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:8001 app:app --timeout 300

# Auto-select root URL based on IS_LOCAL; override with MONGO_MCP_ROOT=... if needed
ifeq ($(IS_LOCAL),true)
  MONGO_MCP_ROOT ?= $(MCP_ROOT_DEV)
else
  MONGO_MCP_ROOT ?= $(MCP_ROOT_PROD)
endif

# ── Ports ───────────────────────────────────────────────────
MCP_PORT   ?= 8000
WEBUI_PORT ?= 8001

# ── AWS credential volume mount ─────────────────────────────
AWS_VOLUME := -v $(HOME)/.aws:/root/.aws

# ── Shared env flags used when running containers ───────────
COMMON_ENV := \
  -e AWS_REGION=$(AWS_REGION) \
  -e IS_LOCAL=$(IS_LOCAL) \
  -e MONGO_CREDS=$(MONGO_CREDS) \
  -e MCP_TOOL_NAME=$(MCP_TOOL_NAME)

MCP_ENV := $(COMMON_ENV)

WEBUI_ENV := \
  -e AWS_REGION=$(AWS_REGION) \
  -e MONGO_CREDS=$(MONGO_CREDS) \
  -e MONGO_MCP_ROOT=$(MONGO_MCP_ROOT)

# ────────────────────────────────────────────────────────────
#  DEFAULT
# ────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

# ────────────────────────────────────────────────────────────
#  BUILD
# ────────────────────────────────────────────────────────────

.PHONY: build build-mcp build-webui

## build        Build both MCP server and WebUI containers
build: build-mcp build-webui

## build-mcp    Build the MCP server container
build-mcp:
	docker build -t $(MCP_IMAGE):latest .

## build-webui  Build the WebUI container
build-webui:
	docker build -t $(WEBUI_IMAGE):latest \
	  --build-context rootfolder=. \
	  webui/

# ────────────────────────────────────────────────────────────
#  PUBLISH  (build + tag + push to ECR)
# ────────────────────────────────────────────────────────────

.PHONY: publish publish-mcp publish-webui ecr-login

## publish      Tag and push both images to ECR
publish: ecr-login publish-mcp publish-webui

## publish-mcp  Tag and push the MCP server image to ECR
publish-mcp: ecr-login build-mcp
	docker tag $(MCP_IMAGE):latest $(MCP_REPO):v$(MCP_VERSION)
	docker tag $(MCP_IMAGE):latest $(MCP_REPO):latest
	docker push $(MCP_REPO):v$(MCP_VERSION)
	docker push $(MCP_REPO):latest

## publish-webui  Tag and push the WebUI image to ECR
publish-webui: ecr-login build-webui
	docker tag $(WEBUI_IMAGE):latest $(WEBUI_REPO):v$(WEBUI_VERSION)
	docker tag $(WEBUI_IMAGE):latest $(WEBUI_REPO):latest
	docker push $(WEBUI_REPO):v$(WEBUI_VERSION)
	docker push $(WEBUI_REPO):latest

## ecr-login    Authenticate Docker to ECR
ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) \
	  | docker login --username AWS --password-stdin $(ECR_REGISTRY)

# ────────────────────────────────────────────────────────────
#  DEPLOY  (ECS force-new-deployment)
# ────────────────────────────────────────────────────────────

.PHONY: deploy-webui

## deploy-webui  Force a new ECS deployment of the WebUI service
deploy-webui:
	aws ecs update-service \
	  --cluster ??? \
	  --service  ??? \
	  --force-new-deployment \
	  --region $(AWS_REGION)

# ────────────────────────────────────────────────────────────
#  RUN DIRECTLY  (no container — uses local venv)
# ────────────────────────────────────────────────────────────

.PHONY: run run-mcp run-webui

## run          Run both MCP server and WebUI directly (background + background)
run: run-mcp run-webui

## run-mcp      Run the MCP server directly with fastapi
run-mcp:
	@echo "==> Starting MCP server on port $(MCP_PORT)..."
	AWS_REGION=$(AWS_REGION) \
	  IS_LOCAL=$(IS_LOCAL) \
	  MONGO_CREDS=$(MONGO_CREDS) \
	  MCP_TOOL_NAME=$(MCP_TOOL_NAME) \
	  FASTMCP_STATELESS_HTTP=1 \
	  fastapi run mongo_mcp.py --host 0.0.0.0 --port $(MCP_PORT)

## run-webui    Run the WebUI directly with Flask dev server
run-webui:
	@echo "==> Starting WebUI on port $(WEBUI_PORT)..."
	AWS_REGION=$(AWS_REGION) \
	  MONGO_CREDS=$(MONGO_CREDS) \
	  MONGO_MCP_ROOT=$(MONGO_MCP_ROOT) \
	  bash -c 'cd webui && $(WEBUI_EXC)'

# ────────────────────────────────────────────────────────────
#  RUN FROM CONTAINER  (using local image built above)
# ────────────────────────────────────────────────────────────

.PHONY: run-mcp-container run-webui-container run-containers

## run-containers         Run both MCP and WebUI from containers (detached)
run-containers: run-mcp-container run-webui-container

## run-mcp-container      Run the MCP server from the local container image
run-mcp-container:
	@echo "==> Running MCP container on port $(MCP_PORT)..."
	docker run --rm -d \
	  --name mcp-server \
	  -p $(MCP_PORT):8000 \
	  $(AWS_VOLUME) \
	  $(MCP_ENV) \
	  $(MCP_IMAGE):latest

## run-webui-container    Run the WebUI from the local container image
run-webui-container:
	@echo "==> Running WebUI container on port $(WEBUI_PORT)..."
	docker run --rm -d \
	  --name mcp-webui \
	  -p $(WEBUI_PORT):8001 \
	  $(AWS_VOLUME) \
	  $(WEBUI_ENV) \
	  $(WEBUI_IMAGE):latest

# ────────────────────────────────────────────────────────────
#  STOP CONTAINERS
# ────────────────────────────────────────────────────────────

.PHONY: stop stop-mcp stop-webui

## stop         Stop both running containers
stop: stop-mcp stop-webui

## stop-mcp     Stop the MCP server container
stop-mcp:
	docker stop mcp-server 2>/dev/null || true

## stop-webui   Stop the WebUI container
stop-webui:
	docker stop mcp-webui 2>/dev/null || true

# ────────────────────────────────────────────────────────────
#  LOGS
# ────────────────────────────────────────────────────────────

.PHONY: logs logs-mcp logs-webui

## logs         Tail logs from both containers
logs:
	docker logs -f mcp-server & docker logs -f mcp-webui

## logs-mcp     Tail MCP server container logs
logs-mcp:
	docker logs -f mcp-server

## logs-webui   Tail WebUI container logs
logs-webui:
	docker logs -f mcp-webui

# ────────────────────────────────────────────────────────────
#  HELP
# ────────────────────────────────────────────────────────────

.PHONY: help
## help         Show this help message
help:
	@echo ""
	@echo "Usage: make [target] [VAR=value ...]"
	@echo ""
	@echo "Overridable variables:"
	@echo "  MCP_VERSION=$(MCP_VERSION)        WEBUI_VERSION=$(WEBUI_VERSION)"
	@echo "  MCP_PORT=$(MCP_PORT)           WEBUI_PORT=$(WEBUI_PORT)"
	@echo "  MCP_TOOL_NAME=$(MCP_TOOL_NAME)  MONGO_CREDS=$(MONGO_CREDS)"
	@echo "  IS_LOCAL=$(IS_LOCAL)           MONGO_MCP_ROOT=$(MONGO_MCP_ROOT)"
	@echo "  MCP_ROOT_DEV=$(MCP_ROOT_DEV)"
	@echo "  MCP_ROOT_PROD=$(MCP_ROOT_PROD)"
	@echo "  AWS_REGION=$(AWS_REGION)       AWS_ACCOUNT=$(AWS_ACCOUNT)"
	@echo ""
	@echo "  Set IS_LOCAL=false to target prod ($(MCP_ROOT_PROD))"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo ""
