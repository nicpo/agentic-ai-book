# Chapter 6: Multi-Agent Systems

## 6.1 Why Multi-Agent?

A single agent with access to all necessary tools can, in theory, handle arbitrarily complex tasks. In practice, single agents break down in predictable ways as task complexity grows:

- **Context window saturation:** A long-running task accumulates so much history that the model loses coherence.
- **Role confusion:** An agent asked to simultaneously be a meticulous researcher, a critical reviewer, and a concise writer performs all three roles poorly.
- **Sequential bottleneck:** When subtasks are independent, a single agent processing them sequentially wastes wall-clock time.
- **Skill mismatch:** A single general model may be adequate but not optimal for every subtask — a code execution task benefits from a model with strong code reasoning; a customer-facing response benefits from a model fine-tuned for tone.

Multi-agent systems address these by decomposing a task across multiple agents, each with a narrower scope, a specialized role, or an isolated context window.

## 6.2 When to Decompose: Single vs. Multi-Agent Criteria

The decision to add agents has real costs: coordination overhead, increased latency, more failure points, and debugging complexity that grows super-linearly with the number of agents. The burden of proof is on the multi-agent design.

**Use a single agent when:**
- The task fits within one context window without degrading
- The task requires tight coherence across all steps (one agent remembers everything it has done)
- Speed matters and the task is inherently sequential
- Debugging and reproducibility are priorities

**Consider multi-agent when:**
- Subtasks are independent and can be parallelized (latency win)
- Different subtasks genuinely require different expertise or prompting styles (specialization win)
- Context windows fill up faster than the task completes (isolation win)
- You need adversarial verification (a reviewer that didn't write the output is more likely to catch errors)
- The system must serve multiple users or tasks simultaneously (concurrency win)

A useful test: "Can I cleanly state in one sentence what each agent is responsible for, with no overlap?" If not, the decomposition is not clean and coordination failures will follow.

## 6.3 Orchestrator-Subagent Architecture

The most common multi-agent pattern. An orchestrator agent (the "supervisor") receives the top-level task, decomposes it into subtasks, delegates each to a specialist subagent, collects results, and synthesizes a final output.

```
                    ┌───────────────┐
                    │  Orchestrator │
                    └───────┬───────┘
                            │ delegates
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │  Research     │ │   Analysis    │ │   Writing     │
   │  Agent        │ │   Agent       │ │   Agent       │
   └───────────────┘ └───────────────┘ └───────────────┘
```

The orchestrator is typically a more capable model (larger, more expensive) while subagents may be smaller specialized models. The orchestrator need not know the implementation details of each subagent — it interacts with them through a defined interface (e.g., each subagent is callable as a tool).

**Routing:** The orchestrator may route tasks based on keywords, intent classification, or its own reasoning. A router-based orchestrator explicitly classifies the input and sends it to the appropriate specialist (e.g., a customer service orchestrator routes billing questions to a billing agent and technical support questions to a technical agent).

## 6.4 Pipeline vs. Reactive Orchestration

**Pipeline orchestration:** Subtasks execute in a fixed sequence. Agent A's output is Agent B's input, which feeds Agent C. This is deterministic and easy to monitor. The weakness: if the task requires dynamic replanning (Agent B's output reveals that the plan was wrong), a fixed pipeline cannot adapt.

```
Input → [Agent A] → [Agent B] → [Agent C] → Output
```

**Reactive orchestration:** The orchestrator observes the output of each step and decides dynamically what to do next. If Agent B's analysis reveals a missing data source, the orchestrator can invoke a data retrieval agent before proceeding to Agent C. This is more flexible but more complex to implement and debug. Example: a research orchestrator routes to a fact-checking agent only when the analyst agent flags a claim as unverified, skipping that step entirely for tasks where nothing was flagged.

**Event-driven orchestration:** Agents emit events; other agents or the orchestrator react to those events. Decoupled, scalable, but harder to reason about ordering and completeness. Better suited to long-running, asynchronous workflows (e.g., monitoring agents that react to alerts) than to tight reasoning chains. Example: a "deployment failed" event emitted by a CI agent triggers a rollback agent and, independently, a notification agent — neither knows the other exists or is reacting to the same event.

## 6.5 Multi-Agent Debate and Verification

A key advantage of multi-agent over single-agent is the ability to use adversarial dynamics for quality improvement.

**Multi-agent debate:** Two or more agents independently generate answers to the same question, then exchange outputs and argue for or against each other's positions over multiple rounds. The final answer is synthesized from the converged position. Effective for tasks with clear correctness criteria (math, logic, factual claims); less effective for subjective tasks. Empirically, this effectiveness as of mid-2026 is mixed, so don't treat it as a default that reliably improves factuality. Instead, compare this with a single-agent baseline.

**Generator-critic pattern:** One agent generates an output; a separate critic agent evaluates it against specific criteria (correctness, completeness, safety, tone) and returns structured feedback. The generator revises based on feedback. More directed than debate; easier to operationalize.

**Independent verification:** For high-stakes tasks (medical decisions, financial calculations, code that deploys to production), have a second agent independently perform the same task and compare outputs. Flag disagreements for human review.

## 6.6 Role Specialization

Effective multi-agent systems assign roles that are genuinely distinct, not just cosmetically different. Common role types:

| Role | Responsibility | Characteristics |
|---|---|---|
| **Researcher** | Retrieve, aggregate, and summarize information | High recall, broad tool access, verbose output OK |
| **Analyst** | Reason over structured data, identify patterns | Strong analytical prompting, structured output |
| **Critic / Reviewer** | Evaluate output for accuracy, completeness, safety | Separate context from the generator, skeptical framing |
| **Executor** | Carry out concrete actions (write file, call API, deploy) | Narrow action scope, conservative on ambiguity |
| **Planner / Orchestrator** | Decompose tasks, delegate, synthesize | High capability model, goal-oriented prompting |
| **Summarizer** | Compress outputs for injection into other agents' contexts | Context-management focused, lossless compression |

The key discipline: give each role the minimum tool access it needs. An executor should not have planner-level reasoning permissions; a summarizer does not need write access to external systems.

## 6.7 Coordination Failure Modes

Multi-agent systems introduce failure modes that do not exist in single-agent systems.

**Deadlock:** Agent A is waiting for Agent B's result before proceeding; Agent B is waiting for Agent A. Solution: timeouts, explicit dependency graphs enforced by the orchestrator.

**Redundant work:** Two agents independently execute the same subtask because the orchestrator failed to track completion state. Solution: a shared task registry that marks work as claimed before it is started.

**Conflicting state:** Two agents write to the same external resource (a document, a database record) concurrently, producing inconsistent state. Solution: serialized writes through a single executor agent, or optimistic locking at the resource level.

**Cascading hallucination:** Agent A hallucinates a fact. Agent B reads Agent A's output and treats it as ground truth. Agent C synthesizes from B's output. By the end, the hallucination is buried in layers of synthesis and hard to trace. Solution: require agents to cite sources and pass provenance metadata downstream; have the final synthesizer verify against primary sources.

**Silent failure propagation:** An agent fails to complete its task but returns a plausible-looking partial result rather than an error. Downstream agents consume the partial result without knowing it is incomplete. Solution: explicit completion status fields in all agent outputs, not just returned text.

**Role confusion:** An agent is given a vague description and interprets its role differently than intended. Solution: precise role descriptions, few-shot examples in each agent's system prompt, role-specific evaluation.

## 6.8 Comparison: Multi-Agent Patterns

| Pattern | Structure | Parallelism | Best For | Key Failure Mode |
|---|---|---|---|---|
| **Orchestrator-subagent** | Hierarchical | Yes (orchestrator dispatches in parallel) | Task decomposition with clear subtasks | Orchestrator bottleneck; bad decomposition |
| **Router-specialist** | Hub-and-spoke | No (one path per request) | High-volume classification-and-dispatch | Misclassification sends to wrong specialist |
| **Pipeline** | Sequential chain | No | Fixed-step workflows | Rigid; error propagation |
| **Generator-critic** | Two-agent loop | No | Output quality improvement | Critic may be sycophantic |
| **Debate** | Peer exchange | Parallel generation, then serial exchange | Tasks with clear correctness criteria | Expensive; can converge to confident wrong answer |
| **Event-driven** | Decoupled | Yes (event-triggered) | Async monitoring, alerts, long-running workflows | Hard to reason about ordering; complex debugging |

## 6.9 Test Yourself

**Q6.1** A product manager proposes building a multi-agent system for every new AI feature. What questions would you ask to evaluate whether multi-agent is warranted vs. a single agent?

**Q6.2** Describe the orchestrator-subagent pattern. What does the orchestrator's system prompt typically contain, and what does each subagent's system prompt contain?

**Q6.3** You have a pipeline where Agent A (researcher) feeds Agent B (analyst) feeds Agent C (writer). Agent B hallucinates a statistic. How does this propagate through the pipeline, and how would you architect the system to catch it? (This is the cascading-hallucination theme, revisited deliberately across the book — see also Q6.8 and, from the failure-mode angle, Q9.1.)

**Q6.4** Explain the generator-critic pattern. Give a concrete example and describe how you would evaluate whether the critic is actually improving output quality vs. just adding latency.

**Q6.5** What is the difference between pipeline and reactive orchestration? Give a scenario where reactive orchestration is required and a fixed pipeline would fail.

**Q6.6** Two agents write to the same shared document concurrently and produce conflicting edits. Describe two ways to prevent this at the architecture level.

**Q6.7** A multi-agent system completes a research task in 20 minutes single-threaded and 5 minutes with 4 parallel subagents. The parallelized version costs 3x more. How would you decide which to deploy?

**Q6.8** What is cascading hallucination in a multi-agent chain? Describe the architecture-level mitigation you would implement.

**Q6.9** You are designing a code review multi-agent system. Define at least three distinct roles, their responsibilities, their tool access, and how their outputs flow to each other.

**Q6.10** Describe how you would implement multi-agent debate for a legal document analysis task. How many rounds? What does convergence look like? When do you stop?

**Q6.11** What is a shared task registry and why does it prevent redundant work in multi-agent systems? What does it need to contain?

**Q6.12** A colleague claims: "Multi-agent systems are just microservices with language models." How do you respond?

**Q6.13** Compare a supervisor-based multi-agent system with a peer-to-peer collaborative system. What are the structural differences, when would you choose each, and what does each sacrifice?

**Q6.14** Two subagents in a pipeline produce conflicting outputs — one recommends approving a loan, the other recommends denying it. Describe four distinct conflict resolution strategies and the conditions under which each is appropriate.

## Further reading

- Anthropic (2025). "How We Built Our Multi-Agent Research System." https://www.anthropic.com/engineering/built-multi-agent-research-system
- Du, Y., Li, S., et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate". https://arxiv.org/abs/2305.14325
- Park, J., O'Brien, J., et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior". https://arxiv.org/abs/2304.03442
- Wu, Q., Bansal, G., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework". https://arxiv.org/abs/2308.08155
