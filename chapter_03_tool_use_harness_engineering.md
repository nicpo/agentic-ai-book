# Chapter 3: Tool Use & Harness Engineering

## 3.1 What Is Tool Calling?

Tool calling (also called function calling) is the mechanism by which a language model invokes an external function or API as part of its generation. The model does not execute the function — it generates a structured output that describes the call: the function name and its arguments. The surrounding code (the harness) parses this output, executes the function, and returns the result to the model.

From the model's perspective, tools extend what it can do: instead of being limited to text in its training data, it can retrieve live information, write files, run code, query databases, send messages, and interact with any system exposed as a function.

## 3.2 API Mechanics

Modern LLM APIs support tool calling natively. The developer provides a list of tool schemas at request time; the model chooses when and how to call them.

**Request structure (simplified):**

```json
{
  "model": "...",
  "messages": [...],
  "tools": [
    {
      "name": "search_web",
      "description": "Search the web and return the top N results.",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "The search query"},
          "num_results": {"type": "integer", "description": "Number of results to return", "default": 5}
        },
        "required": ["query"]
      }
    }
  ]
}
```

**Model response (when it decides to call a tool):**

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "tool_use",
      "id": "call_abc123",
      "name": "search_web",
      "input": {"query": "Claude context window size 2024", "num_results": 3}
    }
  ]
}
```

The harness then executes `search_web(query=..., num_results=3)` and sends the result back:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call_abc123",
      "content": "[search results here]"
    }
  ]
}
```

This pattern repeats until the model generates a text response instead of a tool call.

## 3.3 Schema Design

The quality of your tool schemas has a large, often underestimated, impact on agent reliability. A vague description causes the model to misuse the tool; an overly broad argument type causes the model to pass invalid values.

**Principles for effective schema design:**

**Be explicit about semantics, not just syntax.** A parameter typed as `string` could mean anything. If the parameter expects an ISO 8601 date, say so. If it expects a SQL-safe identifier, say so.

```json
"start_date": {
  "type": "string",
  "description": "Start date in ISO 8601 format (e.g., 2024-06-15). Must not be in the past."
}
```

**Describe the tool's purpose precisely, including what it does NOT do.** If your `search_database` tool only searches product records and not customer records, say so — otherwise the model will call it expecting customer data and be confused by empty results.

**Use enums to constrain categorical arguments.** Instead of `"status": {"type": "string"}`, use `"status": {"type": "string", "enum": ["open", "closed", "pending"]}`.

**Mark required fields explicitly.** Models can and will omit optional fields, but they should not omit required ones. Make the distinction clear.

**Prefer narrower, purpose-built tools over broad, flexible ones.** A `run_sql(query: string)` tool is maximally flexible but maximally dangerous: the model can write arbitrary SQL. Better to expose `get_user_by_email(email: string)` and `get_orders_by_user_id(user_id: string)` separately.

## 3.4 Structured Output Enforcement

By default, models can generate free-form text that looks like a tool call but isn't valid JSON. This breaks the harness. Structured output enforcement forces the model to produce output conforming to a schema.

**Methods:**

- **JSON mode:** The API guarantees valid JSON output but does not enforce a specific schema.
- **Strict schema enforcement:** The API (or local constrained decoding) guarantees output conforming to a given JSON schema. This eliminates an entire class of harness errors.
- **Retry-on-parse-failure:** The harness catches JSON parse errors and re-prompts the model to fix its output. Fragile but sometimes necessary with older models or APIs that don't support strict mode.

The practical preference order: strict schema enforcement > JSON mode with validation > retry-on-parse-failure. The first is fastest and most reliable; the last is slowest and least reliable.

## 3.5 Harness Engineering vs. Framework Abstraction

The **harness** is the code that:
1. Calls the model API
2. Parses the model's response
3. Dispatches tool calls to the appropriate implementations
4. Handles errors (tool failure, timeout, malformed output)
5. Manages the message history / context window
6. Enforces stopping conditions (max steps, budget limits)
7. Logs everything

**Framework abstraction** (LangChain, LangGraph, CrewAI, etc.) provides these capabilities out of the box, plus additional abstractions for agents, chains, and memory. Concretely, "configure rather than build" means declaring an agent's tools, prompt, and control flow through the framework's API (a graph definition, a chain of declared steps, a role config) instead of hand-writing the loop that calls the model, parses its output, and dispatches tool calls yourself. The framework owns the loop; you supply the pieces it expects.

| Dimension                       | Custom Harness              | Framework Abstraction           |
| ------------------------------- | --------------------------- | ------------------------------- |
| **Time to first working agent** | Days to weeks               | Hours                           |
| **Control over execution**      | Complete                    | Limited by framework design     |
| **Debuggability**               | Direct (your code)          | Indirect (framework internals)  |
| **Observability integration**   | Build it yourself           | Often built in or plug-in       |
| **Upgrade path**                | You manage                  | Framework community manages     |
| **Vendor lock-in**              | None                        | Depends on framework            |
| **Hidden complexity**           | None                        | Significant (magic behavior)    |
| **Best for**                    | Production systems at scale | Prototyping, standard use cases |

The key question: "Do I need to understand every token in and out of this system in production?" If yes, the hidden abstractions in frameworks become a liability. If you are building a prototype or a standard RAG pipeline, the productivity gain from a framework is real. The common failure mode is building the prototype in a framework and only then discovering it can't support production requirements — so prototype in a framework, but evaluate early whether production can stay there or needs a rewrite.

## 3.6 Tool Execution Patterns

**Sequential tool execution:** The model calls tools one at a time, waiting for each result before deciding the next call. Simple, easy to debug, but slow when tools are independent.

**Parallel tool execution:** The model emits multiple tool calls in a single generation; the harness executes them concurrently and returns all results together. Supported by most major APIs. Reduces latency significantly for independent tool calls.

Example: An agent researching three companies can issue three `search_web` calls in a single step rather than three sequential steps.

**Tool composition:** A tool can itself be an agent — a complex capability exposed as a single function to the parent agent. This is the mechanism behind sub-agents: from the orchestrator's perspective, `run_data_analysis_agent(dataset=..., question=...)` looks like any other tool call.

## 3.7 Skill Files and Self-Improving Harnesses

### Skill Files

In production agentic systems, it is common to externalize an agent's operating procedures into dedicated **skill files**: standalone documents (typically Markdown) that contain instructions, tool-use guidelines, formatting requirements, and failure-recovery logic for a class of tasks. Rather than embedding all operating logic in the system prompt, a skill file is retrieved and injected when the corresponding task type is detected.

Skill files are the practical implementation of procedural memory (Chapter 4). They make agent behavior explicit, auditable, and independently versioned. A skill for "handle customer refund requests" lives in its own file, can be reviewed by a product team, and can be updated without touching the core agent harness.

**Skill file components:**
- **Task scope:** Which task types this skill covers and which it does not
- **Step-by-step instructions:** The operating procedure the agent should follow
- **Tool-use guidelines:** Which tools to use at each step and in what order
- **Edge case handling:** Known failure patterns and how to recover from them
- **Output format:** Expected structure of the final response or action
- **External files:** A skill is not limited to prose. It can bundle and reference supporting files — helper scripts, SQL queries, config templates — that the agent reads or executes as part of following the procedure, rather than inlining all logic as instructions.

**Skill file example (abbreviated):**

```markdown
---
name: refund-request
description: Use when a customer requests a refund for an order.
---

## Steps
1. Look up the order via `get_order(order_id)`.
2. Check refund eligibility against `policy/refund_rules.sql`.
3. If eligible, call `issue_refund(order_id, amount)`.
...

## Edge cases
- Order not found: ask the customer to confirm the order ID.
...
```

**Discoverability**: A skill's name and description are retrieved the same way tool retrieval works (3.8) — matched against the current task and surfaced when relevant, no explicit invocation needed. What gets "invoked" differs from a tool call, though: instead of executing a function with arguments, the harness injects the skill's instructions into context, and the agent follows them using whatever tools they reference. A `refund-request` skill described as "use when a customer requests a refund" is pulled in the same way an eligible tool would be, but what lands in context is the refund procedure itself, which the agent then carries out via `get_order` and `issue_refund`.

### Self-Improving Harnesses and Loop Engineering

Manual skill optimization is slow and fragile — fixing a failure on Task A often regresses Task B. Self-improving harnesses (e.g., SkillOpt, Microsoft Research) automate this via a text-space optimization loop: the agent runs a batch of tasks and records traces, a verifier scores each trace against a measurable success criterion, an LLM optimizer analyzes failures for recurring patterns, and it proposes bounded edits (add/delete/replace a passage) to the skill file that are validated on a held-out set before being promoted — rejected if they regress other cases. This requires a verifiable feedback signal and a clean held-out set to work at all, so it cannot function on open-ended or subjective tasks. **Loop engineering** is the broader paradigm this sits within: designing the meta-system (evaluation metrics, failure memory, bounded/reversible optimization, exit conditions) that lets an agent safely iterate on its own skill files — while the evaluation design and loop architecture itself remains a human responsibility.

## 3.8 Tool Retrieval

In a system with a small number of tools (5-15), injecting all tool schemas into every model prompt is practical. At enterprise scale — where an agent platform might expose hundreds of tools covering different departments, databases, and APIs — injecting all schemas simultaneously is infeasible: the schemas alone consume a significant fraction of the context window, increasing cost and degrading the model's ability to select the right tool from a crowded list.

**Tool retrieval** solves this by applying the same retrieval logic used for document RAG to tool selection. Tool descriptions are embedded and stored in a vector index. When a task arrives, the system embeds the user query (or the agent's current reasoning step) and retrieves the top-K most semantically relevant tools, injecting only those schemas into the model's context.

**Implementation:**
1. At setup time, embed the description and example usage of each tool and store in a vector index (alongside the tool's full JSON schema).
2. At inference time, embed the user query (or the agent's current intent, if mid-task) and retrieve top-K tool descriptions by cosine similarity.
3. Inject only those K tool schemas into the prompt. Typical K is 5-10.
4. If the agent calls a tool not in the retrieved set, return a structured error pointing toward the correct tool retrieval step.

**When tool retrieval is necessary:** systems with more than ~20 tools; multi-tenant platforms where different tenants have access to different tool subsets; agents whose tool needs vary significantly by task type.

**Risks:** If the retrieval step returns the wrong tools, the agent cannot complete the task — it literally cannot call the right function because the schema was not provided. Mitigation: generous top-K (retrieve 10 even if only 3 are likely needed), fallback to a "search available tools" meta-tool the agent can call explicitly when it believes a relevant tool exists but isn't in its current context.

## 3.9 Test Yourself

**Q3.1** Write a tool schema for a function that sends an email. Include all fields you would expose to the model and explain your choices. What fields would you deliberately NOT expose?

**Q3.2** A model consistently passes `null` for a required tool argument. What are three possible root causes, and how would you diagnose which one is occurring?

**Q3.3** What is the difference between JSON mode and strict schema enforcement? Give a scenario where JSON mode is insufficient.

**Q3.4** Your harness receives a tool call from the model that references a tool name that doesn't exist in the registered tool set. How should the harness handle this? What should it return to the model?

**Q3.5** Compare sequential and parallel tool execution. Give an example of a task where parallel execution would cut total latency by 60% or more.

**Q3.6** A team wants to expose a `run_sql(query: string)` tool to their agent. You recommend against it. Make the argument, then describe what you would expose instead.

**Q3.7** You are building a harness from scratch. List the six minimum capabilities it must have to reliably support a production agent.

**Q3.8** When would you choose a framework abstraction over a custom harness? What signals suggest you are about to hit the limits of a framework?

**Q3.9** An agent has access to three tools that could each answer a "current stock price" query: a web search tool, a financial data API, and an internal database tool. Describe how the agent should reason about which tool to use. What role does tool description quality play?

**Q3.10** What is tool retrieval and when is it necessary? Describe the implementation and the main failure mode specific to this approach.

**Q3.11** A model calls your `send_email` tool with a syntactically valid JSON object but with the wrong field name (`reciepient` instead of `recipient`). Describe the full validation and recovery flow you would implement, from the moment the harness receives the tool call to the moment the model retries correctly.

**Q3.12** What is a skill file and how does it differ from a system prompt? Describe the self-improving harness pattern that allows an agent to optimize its own skill files across runs. What prerequisite must be in place before this approach is viable?

**Q3.13** (Find the bug.) The harness loop below is meant to run a ReAct agent to completion. In testing, it never terminates on tasks the model can actually finish, and token spend runs away. Identify the bug, explain the mechanism, and give the one-line fix.

```python
def run_agent(model, tools, task, max_steps=20):
    messages = [{"role": "user", "content": task}]
    steps = 0
    while steps < max_steps:
        resp = model.generate(messages, tools=tools)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.tool_calls:
            for call in resp.tool_calls:
                result = tools[call.name](**call.args)
                messages.append({"role": "tool", "content": str(result)})
        elif resp.finish:
            return resp.content
    return "max steps exceeded"
```

## Further reading

- Anthropic (2025). "Equipping Agents for the Real World with Agent Skills". https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Microsoft Research (2026). "SkillOpt: Executive Strategy for Self-Evolving Agent Skills". https://www.microsoft.com/en-us/research/publication/skillopt-executive-strategy-for-self-evolving-agent-skills/
- UC Berkeley Gorilla Team. "Berkeley Function Calling Leaderboard". https://gorilla.cs.berkeley.edu/leaderboard.html
