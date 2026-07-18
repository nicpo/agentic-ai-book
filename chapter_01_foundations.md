# Chapter 1: Foundations

## 1.1 What Makes AI "Agentic"?

Traditional machine learning systems take a fixed input and produce a fixed output. You pass an image to a classifier; it returns a label. You pass a sentence to a sentiment model; it returns a score. The computation is one-shot: no iteration, no tool use, no feedback loop, no decision about what to do next.

A chatbot is slightly more dynamic: it maintains a conversation and responds to user input. But a chatbot is still reactive. It does not decide to go look something up, run a calculation, send an email, or break a multi-step task into subtasks. It waits, it responds, it waits again.

**Agentic AI** refers to systems where the model drives a sequence of decisions and actions toward a goal, rather than producing a single response. The model chooses what to do next — whether that means calling a tool, generating a plan, asking a clarifying question, or declaring the task complete. It operates in a loop, and the output of one iteration feeds the input of the next.

The key properties that distinguish an agent from a chatbot or ML model:

- **Goal-directedness:** The agent is working toward a defined objective, not just responding to a prompt.
- **Multi-step action:** The agent takes a sequence of actions, not a single output.
- **Environment interaction:** The agent can read from and write to external systems (APIs, databases, filesystems, browsers).
- **Feedback integration:** The agent observes the results of its actions and adjusts.
- **Autonomy:** Within some scope, the agent decides what to do — it is not purely reactive.

## 1.2 Taxonomy: Traditional ML, Chatbots, Function Calling, and Agents

These categories exist on a spectrum of autonomy and action scope.

| System Type                     | Decision Making                          | Who Controls the Path    | Action Space                           | Feedback Loop                         | Typical Use Case                                         |
| ------------------------------- | ---------------------------------------- | ------------------------ | -------------------------------------- | ------------------------------------- | -------------------------------------------------------- |
| **Traditional ML model**        | None (inference only)                    | None                     | Single output                          | None                                  | Classification, regression, generation                   |
| **Chatbot**                     | Reactive (respond to input)              | User turns               | Text output only                       | Conversational context only           | Customer service, Q&A                                    |
| **Function calling / tool use** | Single-step: choose one tool per turn    | Model picks once         | Predefined tool set                    | Result returned to model in next turn | Augmenting a single response with live data              |
| **Agent**                       | Multi-step: plan, act, observe, continue | Model decides until done | Arbitrary tool set, possibly recursive | Full action-observation loop          | Complex, multi-step tasks requiring external interaction |

## 1.3 The Agent Loop: Perceive, Reason, Plan, Act, Observe

The canonical agent loop has five phases, which repeat until the agent determines the task is complete (or until an external condition halts execution).

```
┌────────────────────────────────────────────────┐
│                                                │
│   PERCEIVE → REASON → PLAN → ACT → OBSERVE     │
│        ↑                              │        │
│        └──────────────────────────────┘        │
│                                                │
└────────────────────────────────────────────────┘
```

**Perceive:** The agent receives its current context — the task description, prior actions and observations, tool results, memory contents, and any injected context (retrieved documents, user messages, system state). This is the input to the current iteration.

**Reason:** The model processes the context. In practice, this means generating a scratchpad, internal chain-of-thought, or structured reasoning that interprets what has happened so far and what is needed next. Some models produce explicit reasoning tokens (extended thinking); others reason implicitly through their generation.

**Plan:** The agent decides on a next action or a sequence of next actions. This might be explicit (the agent outputs a numbered plan before acting) or implicit (the agent goes directly to an action). The sophistication of planning — whether the agent lays out a multi-step plan upfront or proceeds step-by-step — is one of the main axes along which agent architectures differ.

**Act:** The agent executes an action. In LLM-based systems, this means generating a structured tool call: a function name, arguments, and optional metadata. The *harness* (the code surrounding the model) intercepts this output, executes the tool, and returns a result.

**Observe:** The result of the action is fed back into the agent's context. The agent reads the result and enters the next iteration of Perceive.

This loop continues until:
- The agent issues a special "finish" action with a final answer.
- A maximum step limit is reached.
- The harness detects a failure condition and terminates the loop.

## 1.4 Reasoning Models vs. Agent Loops

There is a distinction between a **reasoning model** and an **agent loop built on top of a model**.

**Reasoning models** perform multi-step internal reasoning before producing their output — this happens inside a single model call, the model thinks for N tokens before generating its answer, and the reasoning isn't exposed as a sequence of tool calls. As of mid-2026, all frontier models are reasoning-capable by default. Reasoning depth is a per-request config setting (an effort or thinking-level, typically none/low/medium/high/max) rather than a separate model you switch to. A common pattern is to run most agent steps at low or default effort and escalate to high effort only for the step that's failing, ambiguous, or safety-critical.

**Agent loops** use a standard generation model (or a reasoning model) as the core, but the multi-step iteration happens externally in code: the harness calls the model, receives a tool call, executes it, feeds the result back, and calls the model again. Each model call may or may not include internal reasoning, but the agentic iteration is driven by the external loop, not by the model's internal token generation.

| Property | Reasoning Model (internal) | Agent Loop (external) |
|---|---|---|
| Where reasoning happens | Inside the model's generation | In the harness, between model calls |
| Tool use | Optional (some support it mid-think) | Central to design |
| Latency profile | One long call | Many shorter calls, summed |
| Ability to branch | Internal only; not externally controllable (the token stream is linear, but the model may explore and backtrack within it) | Yes (orchestrator can fork paths) |
| Visibility into steps | Limited (thinking tokens may be hidden) | Full (each action/observation is logged) |
| Cost model | Pay for thinking tokens | Pay per call, multiplied by iterations |

In practice, a reasoning model can be the cognitive core of an agent loop. The agent loop provides tool access and external iteration; the reasoning model provides high-quality single-step reasoning. This is increasingly common in production systems (e.g., running the planner step at high reasoning effort while leaf-node tool calls run at low effort).

## 1.5 Test Yourself

**Q1.1** What is the minimum set of properties that distinguishes an agent from a chatbot? Give a concrete example of a task a chatbot cannot complete but an agent can.

**Q1.2** Explain the difference between function calling and agency. At what point does adding function calling to a chatbot make it an agent?

**Q1.3** Draw the agent loop and label each phase. For each phase, describe one failure mode specific to that phase.

**Q1.4** A product manager proposes building an "agent" to summarize meeting transcripts. Is this actually an agentic task? What questions would you ask to decide?

**Q1.5** You are building a coding assistant. What properties would make it agentic vs. a standard code-completion model?

**Q1.6** Compare the cost and latency profiles of a reasoning model (single call with extended thinking) vs. an agent loop with 10 steps. Under what conditions does each approach win?

**Q1.7** A system calls a weather API every time it generates a response to include current conditions. Is this an agent? Why or why not?

**Q1.8** Your company's CEO asks what "agentic AI" means and how it differs from "regular AI." Give a one-paragraph, jargon-minimal answer appropriate for a non-technical executive.

## Further reading

- Anthropic (2024). "Building Effective Agents". https://www.anthropic.com/engineering/building-effective-agents
- DeepSeek-AI, Guo, D., et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" https://arxiv.org/abs/2501.12948
- Yao, S., Zhao, J., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models". https://arxiv.org/abs/2210.03629
