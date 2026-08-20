# Memory As A System Primitive: Beyond Context Windows And Vector Stores

Most treatments of AI memory make a category error: they treat memory as
content, text retrieved and pasted into a prompt. Even sophisticated systems
where a model reads and writes file-like memory retain this framing. The more
useful frame is that memory is a system primitive: a typed, addressable,
governed substrate that the runtime queries deterministically, and on which
higher-order behavior is composed rather than prompted.

Context windows are registers: fast, local, and wiped on reset. Vector stores
are raw memory: powerful, but unsafe to program against directly. The layer
that matters sits above both: typed records with encapsulated state, scopes as
access modifiers, strategies as virtual methods with genuine dispatch, and
AI-authored functions as compiled, versioned, permissioned methods.

The system described here delivers that substrate through the [Model Context
Protocol](https://modelcontextprotocol.io). MCP is an ecosystem-level bet that
the boundary between probabilistic models and deterministic systems should use
typed, versioned tool contracts. The same discipline belongs one layer down, at
the memory boundary. An agent does not own the store; it mounts memory as a tool
surface with contracts.

Two interdependent moves follow:

1. Determinism can be reintroduced into a probabilistic system by letting an AI
   create fixed retrieval procedures once instead of improvising them every
   turn.
2. Those procedures, together with the records they act on, form a
   virtualization hierarchy: inheritance and dispatch over raw memory, directly
   analogous to object-oriented programming.

Deterministic functions are safe because memory is typed and scoped. Memory is
programmable at scale because functions virtualize it. Neither works alone.

## One Book Of Business, Many Views

Put three people in one insurance carrier: a field agent, an underwriter, and a
risk analyst, each with an AI assistant. They work from the same claims, policy,
and territory data but ask different questions. Scopes keep private context
private; shared scope becomes the commons where discoveries accumulate.

First, an analyst works out the quirks of a legacy claims estate: pagination,
cryptic loss-cause codes, and fields whose names conceal their actual meaning.
That discovery becomes `claims.find_similar(description, region, date_range)`,
a shared method with a typed signature, stable return contract, and hidden
implementation. The knowledge of how to interrogate a domain is no longer
re-derived each session.

Next, a field agent untangles how claims, policies, and territories relate
through composite keys and zone codes. Those relationships become linked
records in the substrate. When the underwriter needs claims by territory, its
assistant dispatches through the shared strategy instead of rediscovering both
the query mechanics and relationships. If it finds a mapping defect, the
correction enters a new draft version, is validated and published, and later
calls resolve to the fix through most-recent dispatch.

Finally, the analyst asks for front-end impact claims with airbag deployment,
mapped across coastal territories. The assistant composes a hybrid narrative
search, a deterministic join through the stored territory relationship, and a
map-ready projection that renders directly to the screen. The relevance score
from the fuzzy first step travels with the result to control marker intensity;
the map data never needs to pass through the model context.

## The Determinism Problem

A vector store is probabilistic at the point of use. Similarity depends on the
query wording, the embedding model, ranking choices, and changing data. Asking
the model to improvise a fresh retrieval plan every turn adds the same
probabilistic tax repeatedly.

The alternative is to separate authoring from execution. The model may use
creative reasoning once to author a named, read-only, multi-step retrieval
procedure. After that, execution is mechanical. The fuzzy work is quarantined
to the first hop, where it belongs and where it can receive the best retrieval
machinery: semantic and lexical rank fusion, optionally followed by reranking.
The rest of the procedure is explicit.

This quarantines nondeterminism to the first fuzzy hop. That hop deserves strong
retrieval machinery: semantic lookup with a quality embedding model, fused with
lexical search and optionally reranked. A deterministic pipeline processing the
wrong records is still wrong. Everything downstream, including filtering,
joining, ranking, and projection, is a fixed and inspectable pipeline.

Three properties make that separation meaningful.

First, values preserve their types between steps. An identifier extracted from a
search result remains an identifier when it feeds a subsequent filter; it does
not become a string that the next operation has to guess how to interpret.

Second, the procedure is read-only by construction. The definition-time
validator permits the supported query operations and rejects write stages and
code-execution operators. The procedure therefore cannot mutate the memory
substrate that it reads.

Third, non-reconstructable values are first-class. A relevance score belongs to
the output that produced it. A later fetch by identifier cannot recreate that
score, so a correct pipeline carries it forward explicitly rather than assuming
it can be recovered from the stored record.

This yields a useful rule: memory is an index, not the truth. Search is how a
system navigates to evidence; authoritative data is fetched, checked, and cited
at the source. The model holds a pointer and a procedure, not an ever-growing
copy of the world.

Execution being mechanical has another consequence: a pipeline's output
contract includes its destination. The model is a possible consumer, not the
mandatory one. A typed result can flow straight to a screen, file, or another
system, sparing the context window. Cost then scales with the decision rather
than data volume: the model orchestrates intent and steps out of the data path.

## Change Must Be Deterministic Too

One deterministic execution is not enough if the procedure can be changed
arbitrarily. Behavioral artifacts need a lifecycle: draft, published, and
retired states; permissioned, single-step transitions; and append-only versions
with back-pointers.

When a procedure is itself a memory object, audit and rollback become ordinary
memory operations. “Why did behavior change?” is no longer an archaeological
exercise across deployment logs; it is a query over the procedure's version
chain.

The versioning rule is simple: search across all versions, but deliver only the
latest applicable version. Older language can remain valuable for semantic
discovery, while the runtime has a single authoritative behavior to execute.
Recency is a dispatch rule, not merely another ranking feature.

Human feedback grounds the lifecycle. Every successful retrieval, miss,
correction, validation, and promotion passes through use. Trust is not granted
once at authoring time; determinism and trust converge across repeated feedback
cycles, with permissioned lifecycle transitions as the point where a person
keeps a hand on the lever.

## OOP Over The Memory Layer

Raw, vector-indexed, decay-scored records are physical memory. They are useful,
but application code should not have to program against them directly, any more
than ordinary software hand-manages heap offsets. A higher abstraction layer
provides encapsulation, inheritance, and dispatch.

| Object-oriented construct | Memory-system analogue |
|---|---|
| Object | A typed memory record with embedded searchable content and structured payload. |
| Public interface | The searchable, embedded projection of a record. |
| Private state | Structured state intentionally kept out of the embedding. |
| Access modifier | Runtime-enforced scope: shared, agent, user, or session visibility. |
| Virtual method | A strategy that can be selected, inherited, and overridden. |
| Method table | The active tool and strategy set for a domain. |
| Compiled method | A named query function with typed inputs and a stable result contract. |
| Vtable swap | Publishing a newer strategy or function version. |

This mapping is load-bearing rather than decorative. A document is the base
class: it carries AI-readable content and structured payload alongside the
framework's deterministic fields for search, retrieval, scope, and scoring. The
split between searchable embedded projection and never-embedded structured state
is encapsulation: query the interface, act on the state.

Scopes become runtime-enforced access modifiers. Shared, agent-local, user-wide,
and session-local visibility is analogous to public, protected, and private
access. Access control stops being prompt discipline and becomes a property
outside the AI's influence.

Strategies are virtual methods. They encode how to act, dispatch by name or
semantic match, and inherit or override general playbooks. Fallbacks are not
hard-coded escape hatches; they are authored abstractions in the same framework.
The playbook for “I do not have a playbook” is itself a record and strategy.
Semantic dispatch is the quarantined fuzzy hop; everything after selection is
deterministic. Versioning supplies the vtable swap: publication silently
redirects callers while older versions remain searchable.

Authored strategies can also be compiled methods bound to a domain, with typed
signatures, stable return contracts, and institutional knowledge hidden behind
the implementation. Loading a domain is loading a class's method table; the
agent calls `domain.method(args)` without reconstructing the pipeline beneath it.

There are therefore two retrieval regimes. General memories, such as facts,
preferences, and episodic notes, use graded relevance: temporal decay,
importance, and semantic fit should all influence recall. Behavioral artifacts
such as strategies use hard dispatch: once selected, the latest published
version must win. Knowledge can be probabilistic; behavior should not drift.

## One Book Of Business, Many Views

Consider an insurance carrier with a field agent, an underwriter, and a risk
analyst. Their assistants work from the same claims, policies, and territory
data but ask different questions. Scope keeps private context private while a
shared scope becomes the commons where reusable discoveries accumulate.

An analyst's assistant might learn the quirks of a legacy claims estate and
publish a shared `claims.find_similar` procedure. A field agent's assistant may
discover how claims relate to territories through an obscure composite key and
store that relationship as linked records. The underwriter then invokes the
shared strategy rather than rediscovering either piece of logic. If the
underwriter finds an error in the territory mapping, the correction follows the
lifecycle, becomes a new published version, and every subsequent invocation
resolves to the fix.

The resulting flow can have one fuzzy hop followed by deterministic work:

1. Rank-fusion search finds claims matching an ambiguous narrative.
2. A stored procedure joins matched claims through known policy and territory
   relationships.
3. The result is shaped directly for a map or other consumer.

The relevance score from the first step can travel with the result to control
marker intensity. Thousands of map points need not be placed into a model
context window merely because a model helped choose the procedure. The model is
one potential consumer of a typed result, not the mandatory destination.

## Stateless Compute Makes The Pattern Necessary

Horizontally scaled workers are intentionally ephemeral. They do not inherit
in-process objects between turns, so any durable object graph must be
reconstituted from an external substrate on every invocation. That is not a
limitation to hide; it is the pressure that makes memory a primitive.

Scopes, strategies, functions, and version chains live in the store. Each
worker mounts the same typed tool contracts through MCP and resolves the same
shared behavior. Consistency follows without sticky sessions or local caches
acting as a source of truth. Determinism and virtualization make this
reconstitution trustworthy.

## Situating This

[MemGPT](https://arxiv.org/abs/2310.08560) made an operating-system analogy
through memory tiers, paging, and interrupts, but it virtualizes capacity:
moving information across fast and slow memory to simulate a larger context.
This approach virtualizes behavior: dispatch, inheritance, and versioned methods
over the store. They are complementary layers of the same stack.
[Graphiti](https://arxiv.org/abs/2501.13956) builds temporally aware knowledge
graphs that preserve historical relationships. Its bi-temporal model resembles
append-only version chains applied to facts rather than procedures. The
[sleep-time compute](https://arxiv.org/abs/2504.13171) line of work promotes
episodic records toward semantic ones through background consolidation; it is a
useful but ungoverned ancestor of a permissioned lifecycle state machine.

## Conclusion

Object-oriented programming organized abstraction, encapsulation, scope, and
polymorphism over raw memory. It let applications call behavior by name, let
data carry guarantees, reach only permitted state, and resolve a call to the
right implementation at runtime. Memory needs all four before an agent can
program and execute against it.

Strategies are named methods. Versioning is polymorphic dispatch that redirects
callers to the newest implementation while older versions remain searchable.
The enforced boundary between fuzzy and deterministic work keeps the result
inspectable: one nondeterministic hop, then fixed and typed behavior. These are
not a feature list. A method is safe to expose because the memory beneath it is
typed and scoped; memory is programmable because methods dispatch over it.
Remove any layer and the rest collapse.
