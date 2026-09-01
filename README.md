# MongoDB Examples with AI and Vector Search

This repository contains complementary projects that demonstrate MongoDB Atlas
integration with AI services, vector search capabilities, and Model Context
Protocol (MCP) implementations for the MongoDB sample Airbnb dataset. Each
project owns its own setup instructions in its own README; this page is a map
of what lives where and why it matters.

![AgenticArchitecture.png](AgenticArchitecture.png)

## Projects

### [MongoMCP/](./MongoMCP/README.md) — MongoDB MCP Server

A configurable Model Context Protocol server that dynamically loads tool
configurations from MongoDB and exposes vector search, Atlas Search, and
aggregation queries to AI agents. It includes a Flask + React Web UI backed by
Amazon Bedrock. This is the core project in the repository, and the one worth
starting with if you want to see MCP, vector search, and a durable memory
layer working together end to end.

- [MongoMCP/README.md](./MongoMCP/README.md) — architecture, prerequisites, and setup for the server and Web UI.
- [MongoMCP/webui/README.md](./MongoMCP/webui/README.md) — running and configuring the Web UI frontend on its own.
- [MongoMCP/tools/](./MongoMCP/tools/) — `mongosetup.py` and related scripts that seed the `mcp_config` database, collections, and a default agent token.

**Why it matters:** MongoMCP treats AI memory as more than retrieved text
pasted into a prompt. It mounts a typed, scoped, and versioned memory
substrate through MCP, so retrieval procedures can be authored once by an AI
and executed deterministically forever after, rather than reimprovised every
turn. Scopes become runtime-enforced access modifiers, strategies become
virtual methods with genuine dispatch, and authored query functions become
compiled, versioned methods that stateless agents can call by name instead of
rediscovering. The full design is written up under
[`MongoMCP/docs/`](./MongoMCP/docs/):

- [Memory As A System Primitive](./MongoMCP/docs/memory-as-a-system-primitive.md) — the conceptual essay: why memory needs typed records, scopes, versioned strategies, and a determinism boundary between fuzzy search and mechanical execution.
- [Memory Architecture And Decisions](./MongoMCP/docs/memory-architecture.md) — the concrete data model: episodic vs. semantic collections, access scopes, and record fields.
- [Memory Workflow](./MongoMCP/docs/memory-workflow.md) — the operational loop agents follow to recall, ground, and retain knowledge, with sequence diagrams.

**Context engineering, not just prompting:** feeding a model more retrieved
text is not the same as giving it the right context. The determinism problem
in [Memory As A System Primitive](./MongoMCP/docs/memory-as-a-system-primitive.md#part-2-the-determinism-problem)
quarantines nondeterminism to a single fuzzy hop (hybrid search), then hands
off to typed, read-only, inspectable steps, so a result's shape and
destination are known in advance instead of reconstructed by the model on
every turn. That is what keeps context windows small and cost tied to the
decision being made, not the size of the underlying data.

Versioned strategies and query functions also close the loop between humans
and agents. Every retrieval that lands or misses, every correction a user
makes, and every promotion from draft to published passes through use, and the
full version history means "why did behavior change?" is a memory query
instead of an archaeology project. See the [feedback loop
section](./MongoMCP/docs/memory-as-a-system-primitive.md#part-2-the-determinism-problem)
of the essay and the lifecycle rules in [Memory Architecture And
Decisions](./MongoMCP/docs/memory-architecture.md) for how draft, published,
and retired states keep that loop auditable and reversible.

### [mcpclient/](./mcpclient/README.md) — Example MCP Clients

Standalone Python clients that connect to the MongoMCP server (or the prior
`dynamicmcp` server) and AWS Bedrock to process user queries with Claude,
using MCP tool calling. Useful as a minimal reference for how a client
authenticates, discovers tools, and drives a conversation loop against the
server, without the Web UI in the way.

- [mcpclient/README.md](./mcpclient/README.md) — available clients, caching behavior, and usage.

### [ASPvectorize/](./ASPvectorize/)

Scripts for configuring MongoDB Atlas Stream Processing to vectorize incoming
Airbnb documents on the fly, rather than embedding a static dataset up front.
Useful if you want to see how vector embeddings can be produced continuously
from a live stream instead of a one-time batch job.

### [PriorVersions/](./PriorVersions/)

Earlier iterations of the MCP server, Web UI, and search implementations,
kept for reference. Each subfolder has its own README describing what it was
exploring; start with the current [MongoMCP/](./MongoMCP/README.md) project
unless you specifically need the history.

## License

See [LICENSE](./LICENSE) file for details.
