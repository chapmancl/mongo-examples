FROM packages.lmig.com/image-lm-base/python3-ubuntu-mutable:latest

ARG CACHE_BUST=675843

ENV NODE_VERSION 22.14.0
ENV NVM_DIR /usr/local/nvm
#RUN apt-get update && apt-get install -y curl
WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt requirements.txt
COPY agent_instructions.md .
COPY *.py ./
COPY mongomcp/ ./mongomcp
# Liberty Air certs
COPY certs/ ./certs


# RUN pip install --no-cache-dir "./mongomcp[agent]"
RUN --mount=type=secret,id=netrc,target=/root/.netrc,uid=1000 pip install -r requirements.txt

#COPY --from=frontend-build /build/frontend/dist frontend/dist
# ============================================================================
# REQUIRED: Tool Configuration
# ============================================================================
# ============================================================================

# Name of the MCP tool to load from MongoDB configuration
# This must match a "Name" field in your mcp_config.mcp_tools collection
ENV MCP_TOOL_NAME=gemSearch
ENV QUERY_EMBEDDING_MODEL_ID=voyage-4-lite
ENV EMBEDDING_MODEL_ID=voyage-4


# ============================================================================
# REQUIRED: MongoDB Connection (Choose ONE option)
# ============================================================================
ENV MONGO_URI=gem-exposure-repository-mongo-db-test-pl-0.z6v9s.mongodb.net

# ============================================================================
# OPTIONAL: MongoDB Configuration Database
# ============================================================================
# Database and collection where MCP tool configurations are stored
ENV MCP_CONFIG_DB=ai_config
ENV MCP_CONFIG_COL=mcp_tools
ENV WEBSERVER_PORT=8001
ENV MONGO_MCP_ROOT=https://gem-mongodb-mcpserver-development.lmig.com

# ============================================================================
# OPTIONAL: AWS Configuration (for Bedrock LLM)
# ============================================================================
# AWS region for Bedrock service
ENV AWS_REGION=us-east-1
# AWS credentials (if not using AWS CLI profile)
#ENV AWS_BEARER_TOKEN_BEDROCK="NOT_THERE"
# Set the command to run the application (replace with your command)
#CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--chdir", "webui", "app:app"]
#CMD ["fastapi", "run", "mongo_mcp.py"]
RUN echo $CACHE_BUST && env


RUN echo `pwd`

RUN echo $CACHE_BUST && ls -la

#  NOW Install Node env
RUN <<EOF cat >> /etc/apt/apt.conf.d/99timeouts
Acquire::http::timeout "240";
Acquire::https::timeout "240";
EOF

RUN --mount=type=secret,id=apt_auth,target=/etc/apt/auth.conf.d/arti.conf \
  export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
  && apt-get -y update \
  && apt-get -y install curl \
  #&& apt-get -y install build-essential \
  #&& apt-get -y install make \
  #&& apt-get -y install rsync \
  && apt-get install -y ca-certificates curl
  #&& rm -rf /var/lib/apt/lists/* \
  #&& groupadd --gid 2000 storage \
  #&& groupadd --gid 3000 node && useradd --no-user-group --uid 3000 -G node,storage --shell /bin/bash --create-home node

RUN --mount=type=secret,id=apt_auth,target=/etc/apt/auth.conf.d/arti.conf \
  buildDeps='xz-utils' \
  && set -x \
  && apt-get clean \
  && apt-get install -y $buildDeps --no-install-recommends \
  && rm -rf /var/lib/apt/lists/* \
  # Install nodejs
  && curl -O https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz \
  && tar -xJf "node-v$NODE_VERSION-linux-x64.tar.xz" -C /usr/local --strip-components=1 \
  && rm "node-v$NODE_VERSION-linux-x64.tar.xz" \
  && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
  && apt-get purge -y --auto-remove $buildDeps
  # Allow node user to see the /urs dir with all its sub directories
  #&& chown -R node:node /usr
  # Create app source directory and giver user permissions
  #&& mkdir -p /home/src/app \
  #&& chown -R node:node /home 

RUN echo "NODE Version:" && node --version
RUN echo "NPM Version:" && npm --version

ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH $NVM_DIR/v$NODE_VERSION/bin:$PATH

# extra CAs, including private CA and Zscaler etc
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# --- Biolerplate above

# Set the working directory inside the container
WORKDIR /app/frontend

COPY frontend/package*.json ./
  
#RUN npm ci oldargs:  --omit=dev --ignore-optional --ignore-scripts
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,uid=1000 \
    npm ci

COPY frontend/ ./

#RUN npm run build
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,uid=1000 \
    npm run build

    #FROM packages.lmig.com/image-lm-base/python3-ubuntu-mutable:latest

    # Set the working directory inside the container
WORKDIR /app

EXPOSE 8001

CMD ["-m", "gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8001", "app:app", "--timeout", "300"]
