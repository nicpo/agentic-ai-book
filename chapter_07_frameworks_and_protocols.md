# Chapter 7: Frameworks and Protocols

## 7.1 The Abstraction Landscape

Building agentic AI systems from scratch requires implementing the agent loop, tool dispatch, state management, error handling, and observability. Frameworks provide these capabilities as reusable abstractions, and protocols standardize how components communicate.

The landscape, as of mid-2026:
- **LangGraph:** State-machine and graph-based agent orchestration (LangChain ecosystem)
- **Model Context Protocol (MCP):** Anthropic's open protocol for standardizing how models connect to tools, data sources, and capabilities (model-to-tool)
- **Agent2Agent (A2A):** an open protocol for standardizing how autonomous agents discover and communicate with each other (agent-to-agent)
- **Vendor agent SDKs:** first-party toolkits (e.g., the OpenAI Agents SDK / Responses API and equivalents from other model vendors) that package the agent loop, tool calling, and handoffs
- **AutoGen / CrewAI / others:** Higher-level multi-agent frameworks with opinionated role definitions
- **Custom harnesses:** First-principles implementations, built to fit specific production requirements

## 7.2 LangGraph

LangGraph is a framework for building stateful, multi-step agents and multi-agent systems using a graph representation where nodes are computations (model calls, tool executions) and edges are transitions between them.

**Core concepts:**

**State graph:** The agent's execution is modeled as a directed graph. Each node receives the current state, performs some operation (call a model, execute a tool, run a Python function), and returns an updated state. Edges define which node to execute next, possibly conditionally based on the current state.

**Checkpointing:** LangGraph can save the state graph at each node, enabling:
- Resumable agents: if an agent fails mid-task, it can restart from the last checkpoint rather than from the beginning
- Human-in-the-loop: the graph can be paused at a designated node for human review or approval before proceeding
- Time-travel debugging: replay any past state to reproduce a bug

**Conditional edges:** A node can return different next nodes based on its output. This enables branching: "if the tool call failed, go to the error handler; if it succeeded, continue to the analysis node."

**Example: a research agent in LangGraph**

```
[retrieve_docs] → [analyze_relevance] → (relevant?) 
    YES → [extract_key_facts] → [synthesize_answer] → END
    NO  → [refine_query] → [retrieve_docs]  (loop back)
```

This is a simple two-branch graph with a loop. LangGraph handles state passing, checkpointing, and conditional routing with minimal boilerplate.

**When to use LangGraph:**
- You need pausable, resumable agents with human approval gates
- Your agent has complex branching logic that is cleaner to express as a graph than as nested if-else in a custom harness
- You are already in the LangChain ecosystem
- You need built-in persistence and state management

**When LangGraph is a poor fit:**
- You need full control over every API call (LangGraph's abstractions can be opaque)
- Your production requirements are not well-served by LangGraph's persistence backends
- Your team does not use Python
- You have latency requirements that LangGraph's overhead violates

## 7.3 Model Context Protocol (MCP)

MCP is an open protocol (released by Anthropic in late 2024) that standardizes how AI models connect to external tools, data sources, and services. Before MCP, every tool integration was custom: developers wrote bespoke code to connect one model to Slack, another to a database, a third to a file system. MCP defines a standard interface so that a tool built for one model works with any MCP-compatible model.

**Architecture:**

```
┌──────────────┐     MCP Protocol      ┌──────────────┐
│  MCP Client  │◄─────────────────────►│  MCP Server  │
│  (AI model + │                       │  (Tool/data  │
│  harness)    │                       │  source)     │
└──────────────┘                       └──────────────┘
```

**MCP primitives:**
- **Tools:** Functions the model can call (web search, file read/write, API calls)
- **Resources:** Data sources the model can read (files, database records, live data feeds)
- **Prompts:** Reusable prompt templates exposed by the server

**Why MCP matters for agents:** An agent that uses MCP-compatible tools can be connected to any MCP server without custom integration code. The ecosystem effect: as more tools publish MCP servers (GitHub, Slack, databases, browsers), agents gain access to them without per-integration work from the agent developer.

**MCP vs. custom tool calling:**

| | Integration effort | Ecosystem | Portability | Standardization | Complexity |
|---|---|---|---|---|---|
| **Custom Tool Calling** | Per-tool custom code | Closed (your tools only) | Tied to your implementation | None | Low for one tool, high for many |
| **MCP** | Write once against MCP spec | Open (any MCP server) | Any MCP-compatible model | Defined protocol | Higher upfront, lower at scale |

## 7.4 Agent-to-Agent Protocols and Vendor SDKs

MCP standardizes the connection between a model and its tools. It does not say anything about how one autonomous agent talks to another. Two layers sit next to MCP.

**Agent2Agent (A2A).** A2A is an open protocol (originated by Google, subsequently moved to open governance) for agent-to-agent interoperability. Where MCP is model-to-tool, A2A is agent-to-agent: it lets one agent discover another's capabilities and delegate work to it across organizational and vendor boundaries. The core concepts:

- **Agent Card:** a published description (typically JSON at a well-known URL) advertising an agent's skills, endpoints, and auth requirements, so other agents can discover it.
- **Tasks:** a client agent sends a task to a remote agent; the remote agent works it, optionally streaming progress, and returns artifacts.
- **Opaque execution:** the remote agent is treated as a black box. The client does not need its prompts, tools, or memory — only its interface. This is what makes cross-vendor delegation possible without exposing internals.

The mental model: **MCP gives an agent hands (tools); A2A gives it colleagues (other agents).** They compose — an A2A-exposed agent can itself use MCP tools internally.

**Vendor agent SDKs.** Beyond open protocols, each major model vendor ships a first-party SDK that packages the agent loop, tool calling, structured output, and multi-agent handoffs (for example, the OpenAI Agents SDK layered on the Responses API, and the equivalent agent toolkits from other vendors). The tradeoff:

- **Pro:** the vendor SDK tracks its own model's tool-calling format exactly, ships sane defaults for retries/streaming/tracing, and reduces boilerplate versus a hand-rolled loop.
- **Con:** it couples you to one vendor's abstractions and release cadence, can lag when you need to swap models, and hides the exact request/response you may need for compliance or cost control — the same "framework vs. custom harness" tension covered in the next section, one level up.

A defensible 2026 default: prototype on a vendor SDK or a lightweight framework, standardize tool access on MCP so tools are portable, expose your agent over A2A only when other teams or vendors actually need to call it, and drop to a custom harness on the paths where control matters (see 7.5).

## 7.5 Custom Harnesses: When to Build vs. Adopt

Here, "framework" means an agent framework — LangChain, LangGraph, CrewAI, AutoGen — that provides the agent loop, state management, and tool dispatch as a reusable library you configure. A **custom harness** is the alternative: you write that loop yourself. Concretely, this is a script or service that (1) calls the model API with the current messages and tool schemas, (2) parses the response for a tool call, (3) dispatches the call to your own internal function, (4) appends the result to the message history, (5) logs the exchange, and (6) repeats until a stop condition is hit — with no framework code in between. For example, a support-ticket agent might use a 100-line Python loop around the Anthropic SDK instead of LangGraph, giving the team full visibility into every prompt and tool call at the cost of building checkpointing and retries themselves.

The framework vs. custom harness decision recurs at every scale. A framework gets you to a working prototype faster; a custom harness gives you control at the cost of build time.

**Signs you should adopt a framework:**
- You are prototyping or exploring capability — time to working demo is the constraint
- Your use case is well-served by the framework's assumptions
- Your team will invest in learning the framework and use it across multiple projects
- You have limited engineering resources and need to leverage existing infrastructure

**Signs you should build custom:**
- You have specific latency, cost, or reliability requirements the framework cannot satisfy
- You need detailed observability at a granularity the framework does not expose
- The framework's abstractions are leaking in ways that cost more time to debug than building from scratch would take
- You are at production scale and framework overhead is measurable
- You need to control every token in and out of your system for compliance or auditability

**A pragmatic middle path:** Use a framework to establish the architecture and identify requirements, then rewrite the critical path in a custom harness once requirements are stable. The prototype revealed what you actually need; the production system builds that and nothing else.

## 7.6 Comparison Table: Framework Options

Landscape as of mid-2026 (the framework ecosystem moves fast, so verify current production-readiness and API surface before relying on this table).

| Framework | Abstraction Level | Best For | Weaknesses | Adoption as of mid-2026 |
|---|---|---|---|---|
| **LangGraph** | State graph | Complex branching agents, human-in-the-loop, resumable tasks | LangChain dependency, opaque internals | Widely deployed in production |
| **CrewAI** | Role-based multi-agent | Quick multi-agent prototyping with defined roles | Limited flexibility, less control | Common for prototypes; less common in hardened production |
| **AutoGen** | Conversational multi-agent | Research, experimentation, debate patterns | Less production-hardened | Mostly research and experimentation |
| **Vendor agent SDK** | Vendor-opinionated loop | Fast build against a specific model, first-party tracing/handoffs | Vendor coupling, harder model swaps | Growing quickly; tied to each vendor's release cadence |
| **Custom harness** | None | Full control, production requirements | Build time, maintenance burden | Common on control-critical paths at scale |

## 7.7 Test Yourself

**Q7.1** Explain what LangGraph's checkpointing feature enables. Give two production use cases where checkpointing is important.

**Q7.2** What problem does Model Context Protocol solve? Without MCP, what does a developer have to do to connect an AI model to a new tool?

**Q7.3** You are building a customer service agent that needs human approval before issuing refunds above $100. How would you implement this in LangGraph?

**Q7.4** A team has built a working prototype in LangChain and now wants to move to production. What questions would you ask to decide whether to stay in the framework or write a custom harness?

**Q7.5** Compare MCP tools, MCP resources, and MCP prompts. Give a concrete example of each for a document management agent.

**Q7.6** Describe conditional edges in LangGraph. Give an example where a conditional edge prevents unnecessary computation.

**Q7.7** Your team is starting a new agentic AI project. Recommend a starting point (framework or custom harness) and justify your choice based on the team's goal of launching a demo in 2 weeks followed by a production deployment 3 months later.

**Q7.8** Suppose you're asked: "Have you used LangChain? What's your opinion of it?" Give a balanced, technically grounded answer that reflects real experience rather than framework advocacy.

**Q7.9** MCP and A2A are both "standardization" protocols, yet they solve different problems. Explain the distinction, give one concrete scenario where you would reach for A2A and not MCP, and describe how the two compose in a single system.

**Q7.10** A team wants to build a new agent and is debating between a vendor's first-party Agents SDK and a hand-rolled custom harness. Lay out the tradeoffs, then recommend a default path for a system that must (a) ship in a month and (b) remain able to swap the underlying model later.

## Further reading

- Agent2Agent (A2A) Project. "A2A Protocol Specification". https://a2a-protocol.org/
- Anthropic. (2024). "Introducing the Model Context Protocol". https://www.anthropic.com/news/model-context-protocol
- CrewAI Documentation. https://docs.crewai.com/
- LangChain. "LangGraph Documentation". https://docs.langchain.com/oss/python/langgraph/overview
- Model Context Protocol. "Specification". https://modelcontextprotocol.io/
- OpenAI. "OpenAI Agents SDK Documentation". https://openai.github.io/openai-agents-python/
- Wu, Q., Bansal, G., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework". https://arxiv.org/abs/2308.08155
