# Memory Architecture And Decisions

DynamicMCP provides a durable memory layer through MCP tools. It is designed for
stateless agents and workers: each invocation can mount the same typed memory
surface, retrieve the relevant state, and execute against a shared set of
contracts.

## Model

Memory is stored in two MongoDB collections:

| Collection | Role |
|---|---|
| `memory_episodic` | Session-bound and short-lived working context. |
| `memory_semantic` | Long-lived facts, preferences, schemas, strategies, and architecture knowledge. |

Every record can carry a `memory_type`, content, structured payload, tags,
entities, importance, decay rate, ownership metadata, and explicit
`related_docs` graph edges. The embedded content supports semantic retrieval;
the payload holds structured state that callers can query without asking a
model to reconstruct it.

## Access Scopes

Scopes are integer levels with explicit visibility rules. The runtime applies
them independently of the model.

| Scope | Visibility | Typical use |
|---:|---|---|
| `0` | Everyone | Shared strategies, public schemas, reusable knowledge. |
| `10` | One agent | Agent-private operating state. |
| `20` | One user | Durable user preferences and personal knowledge. |
| `30` | One user and session | Conversation working state. |
| `40` | One user, session, and agent | Strictly isolated task state. |

The scope determines both visibility and storage tier: session-bound scopes use
episodic memory, while broader scopes use semantic memory.

## Recall And Graph Navigation

`memory_recall` searches one or both memory tiers. For semantic lookup, the
service combines vector relevance, decayed importance, and recency rather than
ranking solely by embedding similarity. Returned records can expand through
their `related_docs` links using bounded breadth-first traversal, making a
memory graph navigable from a relevant entry point.

The system distinguishes two kinds of retrieval:

- **Recollection** is graded: relevance, importance, and age all matter.
- **Behavior dispatch** is deterministic: once a strategy chain is located,
  the latest version is the authoritative implementation.

This separation lets ordinary knowledge fade appropriately while preventing an
older procedure from silently winning over its replacement.

## Strategies And Versioning

Strategies are long-lived behavioral records. A new version is appended rather
than overwriting history, and the previous version is linked forward. Historical
versions stay semantically searchable because older wording can still be the
best way to locate a concept. After discovery, version resolution delivers the
current version as the single authoritative read head.

This preserves an audit trail and supports questions such as “why did this
behavior change?” without making runtime execution ambiguous. Access frequency
is transferred to the new version so its usage signal survives a version
rotation.

## Deterministic Query Functions

The agent can define a named, multi-step query function for an active data
domain. A function is validated before it is stored and may use supported read
operations such as vector, text, hybrid, geospatial, rerank, and aggregation
searches. Validation rejects write stages and code-execution operators.

Functions are versioned behavioral artifacts with a controlled lifecycle:

```text
prod <-> dev <-> disabled -> delete
```

Each transition moves one step and requires the appropriate permission. The
function builder also verifies that target domains are active, checks collection
membership when live metadata is available, and prevents a function from
overwriting a non-function tool.

## Why This Shape

The architecture separates creative authoring from repeatable execution. A
model may use semantic reasoning to author or select a retrieval procedure, but
the stored procedure executes mechanically against typed records. This makes
recurring work inspectable, shareable, and safe to run across stateless workers.

For the broader rationale, see [Memory as a System Primitive](memory-as-a-system-primitive.md).
For the request lifecycle, record contracts, and MCP examples, see
[Memory workflow](memory-workflow.md).