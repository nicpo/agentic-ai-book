# Quick Reference

*One-page summaries of conceptual frameworks.*

---

## QR-1: Agent Architecture Patterns

| Pattern | Core Mechanism | Planning Horizon | Self-Correction | Parallelism | Best For | Key Weakness |
|---|---|---|---|---|---|---|
| **ReAct** | Interleaved Thought → Action → Observation | Single step | None | No | Short tasks, transparent reasoning | No lookahead; degrades on long tasks |
| **Plan-and-Execute** | Upfront plan, then execute step by step | Multi-step | Replanning on failure | Optional (parallel execution steps) | Long-horizon tasks with stable structure | Plan errors amplify; bad at adaptive tasks |
| **Reflexion** | Attempt → self-critique → revised attempt | Per-attempt | Yes, via stored reflection | No | Tasks with verifiable outcomes (code, math) | Needs reliable self-diagnosis signal |
| **Generator-Critic** | Generator produces; separate critic evaluates | Within task | Yes, via critique loop | No | Output quality improvement | Critic may be sycophantic |
| **Orchestrator-Subagent** | Orchestrator decomposes and delegates | Multi-step | Via replanning | Yes (subagents in parallel) | Complex tasks with independent subtasks | Coordination overhead; cascading failures |
| **Event-Driven** | Agents react to events from other agents or systems | Async | No | Yes | Long-running monitoring, async workflows | Hard to reason about ordering |
| **Deliberative / Lookahead** | Search over future action sequences before acting | Multi-step ahead | Implicit (search prunes bad paths) | No | High-stakes irreversible decisions | Computationally expensive |

**When to choose single vs. multi-agent:** Single agent when the task fits in one context window, requires tight coherence, or speed and simplicity matter. Multi-agent when subtasks are independent (parallelize), require genuinely different expertise, or exceed one context window.

---

## QR-2: Memory Architecture Comparison

| Memory Type | Where Stored | Persistence | Retrieval Method | Latency | Capacity | Best For |
|---|---|---|---|---|---|---|
| **In-context (short-term)** | Context window | Session only | None — always present | 0 ms | Context window limit | Short tasks, conversation history |
| **Episodic (vector store)** | Vector DB | Cross-session | Semantic similarity (ANN) | 10-100 ms | Millions of entries | User personalization, long-horizon task history |
| **Procedural** | Prompt library / vector DB | Permanent | Tag or semantic lookup | 10-100 ms | Hundreds of procedures | Recurring task types, codified workflows |
| **Key-value store** | In-memory DB | Session or persistent | Exact key lookup | <1 ms | Thousands of keys | Task-specific tracking state, counters |
| **Graph (knowledge graph)** | Graph DB | Persistent | Graph traversal / pattern match | 5-50 ms | Millions of nodes/edges | Entity relationships, dependency tracking |
| **Summarized history** | File or DB | Cross-session | Injected at session start | 0 ms (once loaded) | Bounded by summary size | Multi-session tasks where detail loss is acceptable |
| **Relational DB** | SQL database | Persistent | SQL query | 5-50 ms | Effectively unlimited | Structured enterprise data the agent queries |

**Decision rule:** Use in-context for everything that fits. Add episodic memory when you need cross-session recall. Add procedural memory when the same task type recurs. Use key-value for ephemeral per-task state. Use graph memory when the query is relational ("what depends on X?").

---

## QR-3: Evaluation Axes and Metrics

| Eval Axis | What It Measures | Primary Metric | Alert If... |
|---|---|---|---|
| **Objective completion** | Did the agent finish the task correctly? | Task completion rate (pass/fail or partial credit) | Rate drops >5% from baseline |
| **Reasoning integrity** | Was the reasoning sound, step by step? | Step-level accuracy vs. golden trajectory | Divergence rate rises |
| **Operational safety** | Did the agent avoid harmful or unauthorized actions? | Policy violation rate; irreversible action rate | Any meaningful rate |
| **Resource efficiency** | How much did it cost in tokens and dollars? | Cost per task | Cost spikes >2x baseline |
| **Latency** | How long did it take? | P50 / P99 wall-clock time | P99 exceeds SLA |
| **Robustness** | Does performance hold under edge cases and adversarial inputs? | Delta on perturbed eval set vs. clean eval | Delta >5 percentage points |

**Eval method selection:**

| Situation | Method |
|---|---|
| Ground truth exists | Golden trajectory + step-level accuracy |
| No ground truth, task is open-ended | LLM-as-judge with multi-dimension rubric |
| Verifying tool behavior | Tool-call unit tests (mock tool responses) |
| Pre-production validation | Shadow mode (run new agent, don't serve output) |
| Production rollout | A/B test with downstream success metrics |
| Ongoing production health | Sampled human review + distribution monitoring |

**LLM-as-judge failure modes to mitigate:** position bias (randomize option order), length bias (penalize verbosity in rubric), self-enhancement bias (use a different model as judge), inconsistency (average across multiple calls, calibrate against human labels).

---

## QR-4: RAG Pipeline Decision Points

| Decision | Options | When to Choose Each |
|---|---|---|
| **Chunking strategy** | Fixed-size / sentence / recursive text split / semantic / structure-aware / hierarchical | Fixed-size for speed; structure-aware for formatted docs; hierarchical for best precision |
| **Retrieval type** | Dense only / BM25 only / Hybrid (RRF) | Hybrid almost always; dense for semantic queries; BM25 for exact match (IDs, codes, names) |
| **Query transformation** | None / HyDE / decomposition / step-back / multi-query | HyDE for question-document style mismatch; decomposition for multi-part questions; multi-query for recall improvement |
| **Re-ranking** | None / cross-encoder re-ranker | Add re-ranker when precision at top-5 matters and 50-200 ms latency budget exists |
| **Context size** | Top-3 / top-5 / top-10+ | Start at top-5; more chunks adds noise past the point of relevance |
| **RAG type** | Classical / Agentic / Self-RAG | Classical for fixed pipelines; Agentic when retrieval needs are variable; Self-RAG only with a trained model |

**Hallucination mitigation checklist:** source attribution required in output → faithfulness check post-generation → retrieval precision improvement (hybrid + re-rank) → chunk quality audit → temperature reduction → context noise reduction (fewer but better chunks).

---

## QR-5: Security and Guardrail Layers

| Layer | What It Catches | Where It Sits |
|---|---|---|
| **Input classification** | Direct prompt injection, off-topic requests, policy violations in user input | Before the agent loop |
| **Tool argument validation** | Invalid arguments, wildcard paths, missing required fields, out-of-range values | Harness, before tool execution |
| **Tool permission check** | Agent calling a tool it doesn't have access to | Harness, before tool execution |
| **Approval gate** | Irreversible or high-risk actions above defined thresholds | Harness, conditional node before execution |
| **Sandboxing** | Unauthorized filesystem, network, or system access by executed code | Execution environment |
| **Egress filtering** | Data exfiltration via outbound HTTP, API calls to unwhitelisted domains | Network layer |
| **Output filtering** | PII in outbound messages, policy violations in generated content | After generation, before delivery |
| **Rate limiting** | Runaway loops calling dangerous tools at high frequency | Tool implementation |
| **Loop detection** | Infinite loops (same tool + args repeated N times) | Harness, per-iteration check |
| **Kill switch** | Operator halts a running agent immediately | External signal, checked at loop start |
| **Audit log** | Post-hoc attribution and compliance | Persistent, tamper-evident log of every action |

**Prompt injection defense summary:** privilege separation (instructions vs. content are distinct); output monitoring for actions inconsistent with the original task; minimal tool scope (can't be instructed to do what the tool doesn't permit).

---

## QR-6: Production Monitoring Checklist

**Metrics to track (with alert thresholds):**

| Metric | Refresh | Alert Condition |
|---|---|---|
| Task completion rate | 5 min | Drop >5% from baseline |
| Cost per task | Hourly | Spike >2x baseline |
| P99 latency | 5 min | Exceeds SLA |
| Tool call failure rate (per tool) | 15 min | >5% for any tool |
| Loop detection trigger rate | 15 min | >0.1% of sessions |
| Error rate by type | 5 min | New error type appears |
| Token usage per step | Hourly | Approaching per-step limit |
| Human approval queue depth | Real-time | >N tasks waiting |

**Deployment progression:** Offline eval (regression suite) → Shadow mode (production traffic, outputs not served) → Canary (5% of traffic, full monitoring) → Full rollout. Rollback trigger: task completion drops >2 pp, P99 exceeds SLA, or error rate rises >1 pp.

**Debugging a failing agent (decision tree):**
1. Pull full trace for a failing task
2. Identify the step where behavior diverged from expected
3. Was the tool result correct? If no → tool or retrieval problem
4. Did the model correctly interpret the tool result? If no → reasoning/prompt problem
5. Was the necessary context present? If no → memory/context management problem
6. Was there a recent change to the model, tool API, or system prompt? → regression
