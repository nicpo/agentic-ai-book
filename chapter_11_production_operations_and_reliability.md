# Chapter 11: Production Operations and Reliability

## 11.1 The Gap Between Demo and Production

An agent that works in a notebook demo is not the same as an agent that runs reliably in production at scale. The gap is substantial and often underestimated. Production agents face non-deterministic model outputs, flaky external tools, user inputs that differ from the eval distribution, load spikes, and cascading failures. Bridging this gap requires the same operational discipline that production engineering applies to any distributed system — plus additional challenges specific to probabilistic AI components.

## 11.2 Observability: What to Log

For traditional software, you log requests, errors, and latency. For agents, you log the entire reasoning chain: every token that goes into the model, every token that comes out, every tool call and its result, every decision the agent makes.

**What to log per agent step:**
- **Timestamp and step index**
- **Full prompt sent to the model** (system prompt, conversation history, injected context)
- **Full model response** (including any chain-of-thought or reasoning tokens)
- **Tool call name and arguments** (if any)
- **Tool result** (full response, or a hash + reference for large payloads)
- **Step latency** (time from model call start to tool result received)
- **Token counts** (input tokens, output tokens, reasoning tokens if exposed)
- **Model version and parameters** (temperature, max tokens, etc.)
- **Agent session ID and task ID** (for correlation across steps)
- **Error or exception** (if any)

This log is the ground truth for debugging. Without it, a failing agent is a black box.

**Structured logging:** Log in a machine-readable format (JSON) so that you can query, aggregate, and alert on specific fields. Free-form log messages are adequate for debugging individual runs but do not scale to fleet-level monitoring.

**Sampling:** At high throughput, logging every token of every run becomes expensive. Sample a fraction of runs (e.g., 10%) for full logging; log only summaries (step count, final outcome, cost, latency) for the rest.

## 11.3 Distributed Tracing for Agents

An agent run is a tree of operations: a root task spawns tool calls, which may themselves invoke sub-agents, which call more tools. Distributed tracing systems (OpenTelemetry, Langfuse, LangSmith, Honeycomb) represent this tree structure and allow you to drill into any branch.

**Trace elements:**
- **Spans:** Each model call, tool call, or sub-agent invocation is a span with a start time, end time, and attributes.
- **Parent-child relationships:** Tool call spans are children of the model call span that invoked them; sub-agent spans are children of the orchestrator's delegation span.
- **Trace ID propagation:** A single trace ID threads through all spans for a given task, including across service boundaries (when sub-agents run in separate services).

**What you can ask with tracing:**
- Which step accounts for the most latency in a 45-second task?
- Which tool call is failing most frequently?
- How does step count distribution differ between successful and failed tasks?
- Is a specific sub-agent contributing disproportionate cost?

## 11.4 Monitoring and Alerting

**Metrics to track in production:**

| Metric                          | Why It Matters                   | Alert Condition                  |
| ------------------------------- | -------------------------------- | -------------------------------- |
| **Task completion rate**        | Primary product health signal    | Drop > 5% from baseline          |
| **Cost per task**               | Economic viability               | Spike > 2x baseline              |
| **Median and P99 latency**      | User experience                  | P99 > SLA threshold              |
| **Error rate by type**          | Which components are failing     | Error type spike                 |
| **Tool call failure rate**      | Tool reliability                 | > 5% failure for any tool        |
| **Loop detection triggers**     | Agent getting stuck              | Any meaningful rate              |
| **Human approval request rate** | HITL load, unexpected escalation | Significant change from baseline |
| **Token usage per step**        | Context window management        | Approaching per-step limit       |

**Anomaly detection:** Model behavior drift is subtle and may not manifest as a hard error. Monitor distributions: if the average step count per task doubles without a corresponding improvement in completion rate, something has changed.

## 11.5 Debugging a Failing Agent

A systematic approach to agent debugging, by failure category:

**Agent produces wrong final answer:**
1. Pull the full trace for the failing task.
2. Identify the step where the reasoning diverged from the correct path.
3. Check: was the retrieved/tool-provided information correct? If not, the problem is upstream (retrieval, tool).
4. Check: did the model correctly interpret the correct information? If not, the problem is reasoning.
5. Check: was the prompt missing context that would have prevented the error?

**Agent gets stuck in a loop:**
1. Identify the repeating (tool, arguments) pattern in the trace.
2. Check: what observation is the agent receiving? Is it empty, unexpected, or ambiguous?
3. Check: is the agent's reasoning correctly updating given the repeated observation, or is it ignoring the repetition?
4. Fix: update the loop detection logic to inject a "stuck" message earlier; or update the prompt to handle the specific observation pattern.

**Agent produces hallucinated tool calls:**
1. Check: does the tool name exist in the registered tool set? If not, the model is confabulating tools.
2. Check: do the arguments match the schema? If not, the schema may be unclear.
3. Fix: clarify the tool descriptions; add examples of correct calls in the tool schema's description; increase schema strictness.

**Agent is slow:**
1. Identify the longest spans in the trace.
2. Check: is latency in the model call (reasoning latency), the tool call (external API latency), or the harness (overhead)?
3. For model latency: consider a smaller model for simpler steps, or reduce context size.
4. For tool latency: add caching; use faster API endpoints; parallelize independent tool calls.

## 11.6 Cost Management

**Token spend is the primary cost driver.** Track token spend at the task level, broken down by input tokens, output tokens, and (if applicable) reasoning tokens.

**Bounding runaway loops:** A loop-detection budget cap halts the agent if token spend exceeds a per-task maximum. Implement this as a check at the start of each iteration: if cumulative cost > budget, halt gracefully.

**Context window optimization:** The largest context window is not always the most economical. A 100K-token context costs more to process than a 5K-token context. Use context summarization aggressively to keep working context small.

**Model tiering:** Use a large expensive model for planning and reasoning; use smaller, cheaper models for tool-result extraction, classification, and summarization subtasks. (As of mid-2026, examples of a frontier/cheap split within one vendor's family are: Claude Opus vs. Claude Haiku, or GPT Sol vs. GPT Luna.) In a multi-agent system, the orchestrator warrants the largest model; leaf-node subagents often do not. A variant here is **dynamic routing**, which passes the request to the most suitable model. It works best where success is objectively verifiable — passing tests, valid schemas, deterministic parsing — and is much harder to apply to subjective tasks like evaluating open-ended writing. Common routing paradigms: **heuristic** (rule-based matching on keywords or prompt patterns), **learned** (a classifier predicts which model performs best for the detected query — see RouteLLM, Ong et al. 2024, which trains routers from preference data), **cascade** (try a cheap model first, escalate only if a verifier flags a bad result — see FrugalGPT, Chen et al. 2023), and **ensemble/fusion** (run multiple models concurrently and synthesize or select the best output).

**Caching:** Cache model responses for identical prompts (useful for repeated retrieval + generation patterns). Cache tool results for queries that return stable data. LLM APIs increasingly support prompt caching (reuse KV cache across calls with the same system prompt), which reduces input token costs significantly for long system prompts.

## 11.7 Latency vs. Quality Tradeoffs

Every agent architecture choice involves a latency-quality tradeoff. The table below describes common tradeoffs:

| Choice         | Faster / Cheaper             | Higher Quality                   |
| -------------- | ---------------------------- | -------------------------------- |
| **Model size**     | Small model                  | Large model                      |
| **Tool calls**     | Fewer, parallel              | More, sequential (more thorough) |
| **Context window** | Shorter (summarized history) | Longer (full history)            |
| **Re-ranking**     | Skip                         | Apply cross-encoder re-ranker    |
| **Planning**       | Reactive (step-by-step)      | Deliberative (plan-first)        |
| **Verification**   | No second pass               | Generator-critic loop            |
| **Retrieval**      | Top-5 dense-only             | Top-50 hybrid + re-rank          |

The right point on this tradeoff depends on the task. A customer service agent responding to "what are your hours?" does not need a plan-first architecture or a cross-encoder re-ranker. A financial analysis agent producing a due-diligence report does.

**Parallelizing sub-tasks:** Independent subtasks should run in parallel. In a research agent, queries to three different knowledge bases can be issued simultaneously. In a multi-agent system, independent subagents can run concurrently. Identifying the critical path and parallelizing off-path work is one of the highest-leverage latency optimizations.

## 11.8 Streaming vs. Batch Execution

**Streaming:** The agent's output is delivered incrementally as it is generated. Users see partial results in real-time. This is appropriate for conversational interfaces where the user is waiting for a response.

For tool-augmented agents, streaming is more complex: the stream must be interrupted when a tool call is generated, the tool must be executed (which may take seconds), and then generation must resume. Most major APIs support streaming with tool call interruption.

**Batch execution:** Tasks are queued and processed offline, without a user waiting. Appropriate for background tasks (nightly reports, bulk data processing). Batch execution enables higher throughput (no need to optimize for first-token latency) and more aggressive cost optimization (batch model APIs are typically cheaper).

**Hybrid:** For long-running tasks, stream intermediate progress signals to the user ("Researching competitor pricing... analyzing results...") while batch-executing the heavy computation. This provides a responsive experience without requiring true streaming of the agent's internal reasoning.

## 11.9 Knowing an Agent Is Behaving Correctly in Production

The fundamental challenge: there is no oracle in production. You cannot check every task against a correct answer. Proxies for correctness:

**Downstream success metrics:** If the agent is a customer service agent, track whether conversations end with issue resolution (can be rated by a follow-up question or inferred from whether the user contacts support again within 24 hours).

**Human review sampling:** Route a random sample of completed tasks to human reviewers. Maintain a consistent review rubric. Use this to calibrate the automated metrics.

**Model-as-judge at scale:** Apply an LLM judge to a sample of production tasks. Cheaper than human review but biased; calibrate the judge against human labels.

**Anomaly monitoring:** Track metric distributions. If step count, token spend, tool call patterns, or output length distributions shift significantly, something has changed — whether a new user behavior pattern, a model update, or a tool API change.

**Canary deployments:** When deploying a new agent version, route a small fraction of traffic to the new version and compare all tracked metrics. Expand traffic only when the metrics match or improve.

## 11.10 Test Yourself

**Q11.1** What is the minimum set of fields you would log for each agent step in a production system? Justify each field.

**Q11.2** Describe how distributed tracing applies to a multi-agent system where the orchestrator spawns three parallel subagents. What does the trace tree look like?

**Q11.3** You receive a report that a production agent "started giving wrong answers" in the last 4 hours. Describe your debugging process step by step.

**Q11.4** An agent's cost-per-task has doubled over the past week with no change to the code or prompts. What are the most likely causes and how would you diagnose?

**Q11.5** Describe three concrete strategies for reducing agent latency without reducing task completion quality.

**Q11.6** When is batch execution preferable to streaming for an agentic system? Give a specific use case for each.

**Q11.7** What does model tiering mean in the context of a multi-agent system, and how does it reduce cost?

**Q11.8** Design a monitoring dashboard for a production customer service agent. List the metrics you would display, the refresh frequency, and the alert conditions.

**Q11.9** An agent is deployed to process 100,000 documents overnight. It completes 73,000 and fails on the rest with a "context window exceeded" error. What went wrong and how would you fix it for the next run?

**Q11.10** How do you know when an agent is behaving correctly in production when you cannot check every output? Describe the monitoring infrastructure you would build.

**Q11.11** Explain prompt caching and how it reduces cost for agents with long system prompts. Under what conditions does it not help?

**Q11.12** Describe a canary deployment strategy for a new agent version. What metrics would trigger a rollback?

**Q11.13** (Diagnose the regression.) The per-task cost/latency log below is for the same agent version and the same task type, one week apart. Code and prompts are unchanged. Explain the most likely root cause and how you would confirm it.

```
                 last week (baseline)    today (regression)
avg steps/task          6.2                    6.4
avg input tokens/step   3,100                 11,800
avg output tokens/step  420                    440
cache hit rate          0.81                   0.07
avg latency/task        4.9 s                 18.3 s
avg cost/task           $0.021                $0.088
```

## Further reading

- Chen, L., Zaharia, M., Zou, J. (2023). "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance". https://arxiv.org/abs/2305.05176
- LangChain. "LangSmith Observability Documentation". https://docs.langchain.com/langsmith/observability
- Ong, I., Almahairi, A., et al. (2024). "RouteLLM: Learning to Route LLMs with Preference Data". https://arxiv.org/abs/2406.18665
- OpenTelemetry. "Generative AI Semantic Conventions". https://opentelemetry.io/docs/specs/semconv/gen-ai/
