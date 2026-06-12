# AI Engineering Concepts

Personal knowledge base — concepts, decisions, and insights accumulated while
learning RAG and agentic AI patterns.

---
## LLM Fundamentals

### Token
A token is the unit of text an LLM processes — roughly ¾ of a word, or ~4 characters of common English (so 100 tokens ≈ 75 words). "Hello" is typically one token; "I'm" splits into "I" and "m". Try it: https://platform.openai.com/tokenizer

LLMs don't read words — they have a fixed vocabulary of ~100K tokens (words, syllables, letters). A tokenizer chops input text into these pieces and maps each to an ID. Rare words or typos just get built from smaller letter/syllable pieces, like Legos.

Cost is calculated from input + output tokens, with output tokens typically costing more — so an architecture that generates verbose intermediate reasoning has a different cost profile than one producing terse structured output.

### Embeddings, Weights, and Parameters

How these three relate, from smallest to largest:

- **Token** — a piece of text (word/sub-word), represented as an integer ID.
- **Embedding** — a vector of numbers (e.g. 4096 floats) assigned to each token ID. It's the model's learned representation of that token's meaning. The full set of embeddings is just one big lookup table — one row per vocabulary token.
- **Weights / parameters** — the embedding table is *one piece* of the model's weights, but most of the weights live in the transformer layers (attention + feed-forward) that sit between the embedding lookup and the final prediction. "Weights" and "parameters" are the same thing — the billions of tunable numbers, learned during training, that encode everything the model "knows" (grammar, facts, reasoning patterns).

**At inference**, the flow is: token IDs → look up embeddings → pass through transformer layers (the bulk of the weights, doing attention + computation) → predict the next token's probabilities.

**Training vs. loading:** during training, supercomputers adjust these billions of numbers over weeks/months until they converge. Once training finishes, the entire grid of weights is saved to disk as a file (the "model weights" you download). Loading a model means reading that file into RAM/VRAM — the numbers themselves don't change after that (unless you fine-tune).

**Model size vs. dataset:** a 7B-parameter model and a 70B-parameter model can be trained on the *same* dataset; model size and dataset size are independent. Think of the dataset as a fixed encyclopedia, the 7B model as a small notebook, and the 70B model as a large filing cabinet, both summarizing the same encyclopedia. The 70B model has 10x more "slots" to store nuance, rare facts, and finer-grained patterns, which is generally why it performs better, even when reading identical training data.


### Context Window
Context window is like the model's working memory — everything it can "see" at once when generating a response. It includes the system prompt, conversation history, any documents you inject, tool results, and the response being generated. Everything counts against the same limit.

A 200K token context window (Claude) sounds large until you realize a 50-page document is roughly 40K tokens, a long conversation history accumulates fast, and every tool call result gets appended to the context. 

Issues with large context windows:
- "lost in the middle" problem — models attend well to content at the beginning and end of a long context but tend to miss information buried in the middle. A 150K token context doesn't mean 150K tokens of equally-weighted information. This is why chunking and retrieval strategies matter even when content technically fits.
- Context window cost scales with every token in and out. A naive implementation that stuffs the full conversation history into every call gets expensive fast at scale. Strategies to reduce the cost:
  - summarizing conversation history
  - evicting stale information
  - dynamically retrieving just the relevant slice for each task — this is what RAG and knowledge-graph/retrieval layers do (see below)

At large enough scale (e.g. an AI tool processing millions of lines of code per run), no context window — however large — can hold everything. Retrieval isn't just a cost optimization at that point; it's the only way to fit the relevant slice of a much bigger corpus into context at all.

Retrieval only reduces cost if it's *selective*. Pulling the entire employee handbook into context for every question isn't retrieval — it's just moving the same large document into the prompt, with the same cost and lost-in-the-middle problems. The savings come from narrowing scope per query: chunk the handbook into small sections, embed them, and at query time fetch only the few chunks relevant to *this* question (e.g. just the parental leave section, not all 200 pages). Poorly tuned retrieval (chunks too large, top-k too high, no confidence gating) can quietly degrade back into "the whole document anyway."

### Temperature
Temperature controls how deterministic vs. creative the model's outputs are. At temperature 0 the model always picks the highest-probability next token — fully deterministic, same output every time. At temperature 1 it samples more broadly, producing varied and creative outputs. Above 1 outputs become increasingly random and often incoherent.

For Compliance checking, code generation, structured data extraction, classification temperature should be 0 or very low (0.1-0.2). You want the same answer every time, you want the most probable correct answer, and creativity is a bug not a feature. 

Creative writing, brainstorming, generating diverse options, conversational responses — these benefit from higher temperature (0.7-1.0) to avoid repetitive or formulaic outputs.


### Prompts

#### Prompt Caching

Prompt caching (a.k.a. context caching) lets a provider reuse the model's internal computation for parts of the prompt that don't change between calls, instead of reprocessing the full prompt from scratch every time.

**What's actually being cached:** Embedding lookups are trivial (a table lookup) and aren't the bottleneck — the expensive part is the transformer's attention computation across all tokens, which scales with context length. Prompt caching stores the intermediate attention activations ("KV cache") for a static prefix of the prompt. On a later call with the same prefix, the model skips recomputing attention for that part and only processes the new/changed tokens (e.g. the latest user message).

**How to use it:**
- Structure prompts so static content comes first and dynamic content comes last — e.g. system prompt + tool definitions + reference docs (static), then conversation history + user query (dynamic).
- The cache only helps if the prefix is byte-for-byte identical across calls. Changing a tool description or reordering tools invalidates the cache for everything after that point.
- Anthropic's prompt caching: cached tokens cost ~10% of normal input price, with a short TTL (~5 minutes, refreshed on each cache hit).

**When it matters:** agent loops with a large, unchanging tool list and system prompt (re-sent on every turn), RAG apps that pin a long reference document across a multi-turn conversation, or any workflow making repeated calls with a shared static prefix.

---

## RAG (Retrieval-Augmented Generation)

### What RAG Is

RAG grounds an LLM's answer in your own documents instead of relying on what the model memorized during training. Pipeline: **embed → vector search → (rerank) → generate**.

```
query ──► embed ──► vector DB (top-k by similarity) ──► [rerank] ──► LLM (context + query) ──► answer
```

The retrieval side ("R") finds relevant text; the generation side ("G") asks the LLM to answer using only that text. Implemented end-to-end in `rfp-assistant-excel/`.

### Two-Stage Retrieval: Bi-encoder + Cross-encoder

| | Bi-encoder (retrieve) | Cross-encoder (rerank) |
|---|---|---|
| How it scores | Encodes query and document **separately**, compares vectors | Encodes query + document **together**, one forward pass |
| Speed | Fast — documents pre-embedded at ingest time | Slow — must run per query/candidate pair |
| Scale | Thousands of documents | Only the top-k candidates from stage 1 |
| Quality | Mediocre on subtle wording differences | Much higher — model sees both texts at once |

Stage 1 (bi-encoder + vector DB) provides **recall**: cheaply narrow thousands of documents to a handful of candidates. Stage 2 (cross-encoder) provides **precision**: re-score just those candidates accurately. Used together because the cross-encoder alone is too slow to run against a whole knowledge base.

### Confidence Gating

Don't always force-feed the LLM a context. Score the top reranked candidate(s) and only pass them as context if they clear a threshold; otherwise tell the user no good match was found rather than generating from a weak match. The threshold lives on the reranker's score scale (raw logits vs 0–1 sigmoid), so it must be re-tuned whenever the reranker model changes.

### Challenges with RAG

- **Retrieval quality is the bottleneck** — chunking, embedding model choice, and reranking determine whether the right context even reaches the LLM. Bad retrieval → bad answer, regardless of model quality.
- **Context dilution** — too many chunks in the prompt buries the relevant one ("lost in the middle") and increases cost/latency.
- **Staleness** — the vector index must be rebuilt as source documents change; it's a snapshot, not live data.
- **Evaluation is hard** — retrieval quality (did we find the right chunks) and generation quality (did the model use them well) need separate metrics and a labeled golden set.
- **No multi-step reasoning** — classic RAG is single-shot retrieve-then-generate; it can't decide "look up X, then based on that look up Y."
- **"Answer only from context" is an instruction, not a guarantee** — the LLM can still drift or hallucinate.

### RAG vs MCP — When to Use Which

These solve different problems and are commonly combined, not competing choices:

| | RAG | MCP / Tools |
|---|---|---|
| Best for | Unstructured knowledge (docs, past Q&A, policies) where semantic similarity is the only practical lookup | Structured or live data (databases, APIs, file systems) that can be queried precisely |
| How it answers | "Find text similar in meaning to the query" | "Run this exact function and get the exact result" |
| Freshness | Snapshot — index rebuilt periodically | Live — hits the source directly |
| Context size | Depends on chunking/reranking quality | Naturally small — fetch only what's needed for the query |

"Keep the context window small" is solved by **better retrieval** (smaller chunks, reranking, confidence gating), not by replacing RAG with tools — tools don't help when the knowledge is unstructured prose with no precise query. For an RFP assistant, the realistic pattern is both: MCP/tools for "look up this client's contract terms in the database," RAG for "find similar answers we've given to this kind of question before."

---

## MCP (Model Context Protocol)

### What MCP Is

**MCP** is an open protocol (introduced by Anthropic, late 2024) that standardizes how LLM applications connect to external tools and data. Think of it as **USB-C for AI**: instead of every app writing custom glue code for every API, the API is wrapped once in an **MCP server**, and any **MCP client** (Claude Code, Claude Desktop, a Python script, PydanticAI) can plug in and use it.

The architecture is client–server:

```
LLM app (host)
    └── MCP client  ──(protocol: stdio or HTTP/SSE)──►  MCP server
                                                          ├── tools      (functions the model can call)
                                                          ├── resources  (data the model can read)
                                                          └── prompts    (reusable prompt templates)
```

- The **server** exposes capabilities — most commonly *tools*, but also *resources* and *prompts*.
- The **client** connects, asks the server what it offers (`list_tools()`), and relays those to the model. The model never talks to the server directly; the client brokers every call.
- The protocol itself is JSON-RPC over stdio (local servers) or HTTP/SSE (remote servers).

The key insight: MCP doesn't add new model capabilities — tool calling already existed. It standardizes the *plumbing*, so integrations are written once and work everywhere.

### Plain Tool vs MCP Tool

A **tool** (in the LLM sense) is a function the model can call — defined as a JSON schema, the model decides when to invoke it, you execute it and return the result.

An **MCP tool** is the same concept but served over a standardized protocol:

| | Plain tool | MCP tool |
|---|---|---|
| Definition | JSON schema hardcoded in the client | Defined once in the server, auto-discovered by any client |
| Execution | Your client runs the function | The MCP server runs the function |
| Reuse | Copy-paste into every project | Any MCP-compatible client connects and uses it |
| Hosting | In-process with the client | Separate process, can be remote |

Plain tools are fine for one-off scripts. MCP tools are the right choice when you want capabilities reused across projects, teams, or clients.

### Benefits of MCP Tools

1. **Discoverability** — the model discovers available tools at runtime via `list_tools()`. The tool description *is* the documentation for the model; no human needs to read docs and hardcode schemas.
2. **Reusability** — define once in the server, use from any MCP-compatible client (Claude Code, Python client, PydanticAI, LangChain). No copy-pasting JSON schemas.
3. **Separation of concerns** — the server owns the implementation; the client owns the orchestration. Update tool logic without touching client code.
4. **Model-friendly output** — shape return values for LLM consumption (concise, no irrelevant fields) rather than for a frontend.
5. **Composability** — connect to multiple MCP servers and the model sees all tools in a flat list. Mix your own servers with external ones (e.g. DeepWiki) with no glue code.

### Why Companies Publish MCP Servers, Not Agents

Tools are **stateless** — call them, get a result, done. An agent has a loop, memory, and decision-making built in. Sharing a stateful agent is harder to host and forces your orchestration logic on the consumer.

When you expose `get_weather` as an MCP tool, any client can use it however they want — once, in a loop, in parallel, conditionally. If you wrapped it in an agent, you'd be dictating the prompt design, model choice, and loop logic to every client. Companies want to own that layer.

**A2A (Agent-to-Agent protocol)** is the emerging standard for sharing agents the way MCP shares tools. As it matures, expect agent marketplaces the same way MCP server directories exist today.

### When to Wrap Company APIs as MCP Tools

REST APIs are designed for **developers writing code**. MCP tools are designed for **models doing reasoning**. If your clients are building agentic systems, expose MCP tools — the model can discover and call them without any human writing glue code.

A company with APIs like `make_sentence_compliant` and `get_customer` should wrap them as MCP tools, not agents. The client's model decides when to call each tool. Group related tools into logical servers:

```
compliance_server  →  make_sentence_compliant, check_policy, flag_content
data_server        →  get_customer, get_contract, search_documents
```

### When to Use MCP

The deciding question is not "internal vs external" — it's **"is the caller an LLM or a programmer?"**

**Frontend → MCP tools? No.** MCP solves a problem frontends don't have: runtime discovery by a model. A frontend developer knows the endpoints at build time, so discovery buys nothing — and costs a lot:

- **Wrong output shape** — good MCP tools return concise, LLM-friendly text; a UI needs structured JSON with IDs, timestamps, pagination cursors, every field.
- **Lost HTTP semantics** — REST gives per-route caching, CDN, status codes, rate limiting, browser-native auth. MCP is JSON-RPC over a session; none of that applies cleanly.
- **Indirection tax** — tunneling typed API calls through a protocol designed for an intermediary (the model) that isn't there.

**Internal agents → MCP tools? Yes.** If internal agents or LLM apps (including Claude Code/Desktop used by the team) need a capability, MCP earns its keep even with zero external consumers: define once, every agent discovers it; the owning team controls the tool description and output shaping; updates propagate at the next `list_tools()`.

Exception: one LLM app calling one function → skip MCP, use an in-process plain tool.

**The pattern: don't choose.** Keep the service layer as the single source of truth and expose two thin doors into it:

```
frontend / mobile / partner devs ──► REST / GraphQL ──┐
                                                      ├──► service layer
internal & external agents ────────► MCP server ──────┘    (single source of truth)
```

The MCP server is a thin wrapper over the same functions the REST handlers call — it adds model-oriented descriptions and reshapes output for LLM consumption. Each consumer type gets an interface designed for it; business logic lives in one place.

### MCP vs A2A

| | MCP | A2A |
|---|---|---|
| **Purpose** | Connect agents to tools and data | Connect agents to other agents |
| **State** | Stateless (tools) | Stateful (tasks with lifecycle) |
| **Task lifecycle** | Fire and forget | submitted → working → completed / failed |
| **Streaming** | Not built in | Native SSE |
| **Maturity** | Production-ready | Emerging (2025) |

Use MCP for capabilities. Use A2A for orchestrating agents across teams or vendors.

---

## Agent Patterns

### Agent vs Subagent

There is no technical distinction — it is purely about role in the system. The same code can be both.

- **Agent** = runs a loop, has autonomy, decides which tools to call
- **Subagent** = an agent being orchestrated by another agent

A client class is an agent when run standalone. Wrap it in a server and have an orchestrator call it, and that same instance becomes a subagent. The label describes the relationship, not the code.

### MCP Tool vs Agent — When to Use Which

- **Capability** (do X to Y) → MCP tool
- **Workflow** (decide how to apply X, Y, Z to achieve a goal) → agent

If the decision-making and sequencing should belong to the *caller*, expose a tool. If you want to encapsulate the reasoning loop, expose an agent.

### Multi-Agent Pattern with MCP

An orchestrator can treat a sub-agent as just another MCP server:

```
orchestrator (agent client)
    └── agent_server  (port 8052)  ← exposes ask_agent() tool
            └── agent client internally
                    ├── kb_server   (port 8050)
                    └── weather_server (port 8051)
```

The orchestrator only knows `ask_agent` exists — it has no visibility into what tools the sub-agent uses. The agent client class is the same at both levels; only the servers it connects to differ.

### Agent Loop Pattern

The standard agent loop:

1. Call model with tools available
2. If model returns tool calls → execute them, append results to messages, go to 1
3. If model returns a final answer → return it
4. Safety cap: `max_turns` prevents infinite loops

---

## APIs and SDKs

### Chat Completions vs OpenAI Responses API

| | Chat Completions | Responses API |
|---|---|---|
| **Availability** | OpenAI, Anthropic, Google, Mistral (standard) | OpenAI only |
| **State** | You manage the messages list | OpenAI stores history server-side |
| **Built-in tools** | Not available | `web_search`, `file_search`, `code_interpreter` |
| **MCP compatibility** | Yes — MCP tool format maps directly | No |
| **Portability** | Works across all providers | OpenAI-specific |

Use Chat Completions when building MCP-based or multi-provider systems. Use Responses API when you need OpenAI's built-in tools.

### Built-in Tools vs Custom Function Tools (Responses API)

- **Built-in tools** (`web_search`, etc.) — OpenAI executes them internally. You just declare them; no execution code needed. You only check `output_item.type == "web_search_call"` if you want to log it.
- **Custom function tools** — same as Chat Completions: model decides to call them, you execute and return the result.

### PydanticAI and LangChain Under the Hood

Both abstract over Chat Completions (or equivalent). PydanticAI auto-generates JSON tool schemas from function signatures and runs the agent loop for you. The difference is operational convenience, not capability. Understanding the raw pattern first makes frameworks trivial to pick up.

### `async with` and Why It's Needed

```python
async with MCPClient() as client:
    ...
```

This calls `__aenter__` on enter and `__aexit__` on exit (even if an exception is raised). `__aexit__` typically awaits `self.exit_stack.aclose()` to tear down all HTTP sessions — that requires `await`, which only works in an async context. A regular `with` can't do this.

`__aenter__` here just returns `self` (nothing async happens on entry). The `async with` is needed purely because of `__aexit__`. Some classes use `__aenter__` meaningfully — e.g. awaiting until a background task is ready — but that's a different case.

---

## Frameworks

### Raw API vs Framework vs MCP

| | Raw API | Framework (`PydanticAI`) | MCP |
|---|---|---|---|
| Schema definition | Hand-written JSON | Auto-generated from function signature | Defined in server, auto-discovered |
| Agent loop | Manual | Framework handles it | Manual (you build it) |
| Tool execution | Manual dispatch | Framework handles it | Manual dispatch |
| Portability | Provider-specific | Multi-provider | Multi-provider + multi-server |
| Best for | Learning, one-off scripts | Production agents | MCP-integrated systems |

---

## Next Steps (Learning Path)

Based on the AI Engineer 2026 roadmap:

- [x] LLM APIs (OpenAI, Anthropic)
- [x] Tool calling — raw and via MCP
- [x] Agent loop pattern
- [x] Multi-server MCP client
- [x] External MCP servers
- [x] **RAG** — ingestion pipeline, embeddings, vector DB, hybrid search, re-ranking
- [ ] Observability — Langfuse tracing, LLM-as-a-judge evals, cost tracking
- [ ] Deployment — Docker, cloud, CI/CD
- [ ] Multi-agent with A2A protocol (emerging)
