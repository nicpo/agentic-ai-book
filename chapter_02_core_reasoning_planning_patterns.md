# Chapter 2: Core Reasoning & Planning Patterns

## 2.1 Overview

The agent loop defines the structure of iteration. The reasoning pattern defines how the model decides what to do at each step — whether to reason before acting, plan ahead, verify its own outputs, or simply react. Understanding these patterns and their tradeoffs is essential for designing agents that are reliable at task completion.

## 2.2 ReAct: Reason + Act

ReAct (Yao et al., 2022) is the most widely used baseline for single-agent reasoning. The model interleaves two types of output in its generation:

- **Thought:** A natural-language reasoning step that interprets the current situation and decides what to do.
- **Action:** A concrete tool call or response.
- **Observation:** The result of the action, fed back into context.

The cycle is: `Thought → Action → Observation → Thought → Action → Observation → ...`

**Example — customer support agent:**

```
Thought: The user wants to know why their order hasn't arrived. I should look up the order status.
Action: lookup_order(order_id="ORD-4821")
Observation: {"status": "shipped", "carrier": "FedEx", "tracking": "123456789", "expected": "2024-06-28"}

Thought: The order is shipped via FedEx with a tracking number. I should also check if there are any carrier delays.
Action: check_carrier_delays(carrier="FedEx", region="northeast")
Observation: {"delay": true, "reason": "severe weather", "estimated_extra_days": 2}

Thought: There is a weather delay. I now have enough to respond to the user.
Action: respond("Your order shipped via FedEx (tracking: 123456789) and is experiencing a 2-day delay due to severe weather in the northeast. New expected delivery: June 30.")
```

**Strengths:** Simple to implement. Reasoning is transparent (logged thoughts are auditable). Works well when each step is relatively independent.

**Weaknesses:** The model can get stuck in a reasoning loop if a tool fails or returns unexpected output. It does not plan ahead — it decides the next action only after seeing the last observation. Long tasks with many steps accumulate context and degrade.

## 2.3 Plan-and-Execute (Hierarchical Planner)

In Plan-and-Execute, the agent generates an explicit plan upfront, then executes each step, optionally replanning when results deviate from expectations.

**Phase 1 — Planning:** The agent takes the task and produces a structured plan: a numbered list of subtasks, each with a description of what to do and optionally what tool to use.

**Phase 2 — Execution:** A separate executor (which may be a simpler model or the same model with a different prompt) carries out each step, observing results.

**Phase 3 — Replanning (optional):** If a step fails or produces unexpected output, the planner is re-invoked to revise the remaining plan.

**Example:** A research agent tasked with writing a competitive analysis:

```
Plan:
1. Search for the top 5 competitors of CompanyX in the cloud storage market.
2. For each competitor, retrieve their pricing page.
3. Extract storage tier pricing into a structured table.
4. Identify which competitor offers the best price-per-GB at 1TB scale.
5. Draft a 300-word summary with a comparison table.

[Executor proceeds step by step, calling search, scrape, extract tools]
```

**Strengths:** Separates high-level planning from low-level execution, which makes each concern simpler. The plan is auditable and can be shown to users for approval before execution. Handles long-horizon tasks better than ReAct because the model is not reasoning-from-scratch at each step.

**Weaknesses:** If the upfront plan is wrong, execution amplifies the error. Replanning adds latency. Works poorly for tasks where the next step is fundamentally unknowable until the current step completes.

## 2.4 Reflexion / Self-Critique Loops

Reflexion (Shinn et al., 2023) adds a self-evaluation phase to the agent loop. After completing a task (or a subtask), the agent critiques its own output — identifying what went wrong, what was suboptimal, and what it would do differently — and stores this reflection as context for the next attempt.

**Structure:**
```
Attempt 1 → Output 1 → Reflection: "I missed checking the edge case where..."
Attempt 2 (informed by reflection) → Output 2 → Reflection: "Better, but I used the wrong API endpoint..."
Attempt 3 → Final output
```

**Where it adds value:** Tasks with verifiable outcomes (code that either runs or doesn't, answers that can be checked against a known correct answer) benefit most. The model can use the failure signal to reason about why it failed rather than just retrying blindly.

**Distinction from simple retry:** A simple retry resubmits the same prompt. Reflexion adds explicit self-diagnosis to the context, giving the model new information to work with.

**Weaknesses:** If the model cannot accurately diagnose its own failures (which it often cannot), the reflections add noise rather than signal. Can be expensive (multiple full attempts). Does not work well when there is no verifiable signal to reflect on.

## 2.5 Chain-of-Thought vs. Tool-Augmented Reasoning

**Chain-of-thought (CoT)** prompts the model to reason step-by-step in natural language before giving a final answer. This improves performance on tasks that require multi-step reasoning (math, logic, decomposed text tasks). One explanation is that this forces the model to externalize intermediate steps, which acts as a form of working memory. Another is that the extra tokens simply buy more serial computation at inference time. The mechanism is not settled, but the accuracy gain on multi-step tasks is robust.

**Tool-augmented reasoning** is CoT where some steps are replaced by tool calls. Instead of asking the model to "calculate 14% of 8,392," you let it call a calculator tool. Instead of asking it to recall recent news, you let it call a search tool.

The insight is that CoT and tool use target different failure modes:
- CoT addresses the model's tendency to compress reasoning steps and make inference errors.
- Tool use addresses the model's lack of real-time information, precise computation, and external state access.

| | Where reasoning lives | Handles real-time data | Handles precise math | Adds latency | Increases cost | Improves accuracy on complex tasks |
|---|---|---|---|---|---|---|
| **Chain-of-Thought** | In-context token generation | No | Partially (error-prone) | Proportional to reasoning tokens | Yes (more tokens) | Yes (significantly) |
| **Tool-Augmented Reasoning** | Combination of tokens + external execution | Yes | Yes (calculator tool) | Proportional to tool call RTT | Yes (tokens + tool overhead) | Yes (more so for certain task types) |

A caveat that matters once you log and evaluate these traces (Ch. 8, Ch. 11): a model's stated chain of thought is not guaranteed to reflect the computation that produced its answer. Studies show models can reach an answer while the verbalized reasoning omits the real cause, or keep emitting reasoning after effectively committing. Treat the trace as useful evidence, but not necessarily as ground truth about the model's internal process.

## 2.6 Reactive vs. Deliberative Planning

**Reactive planning** (also called greedy or step-by-step planning) decides the next action based only on the current state. The model does not look ahead. This is the default for ReAct: you see the current observation, you decide the next action.

**Deliberative planning** involves reasoning about future states before committing to an action. The model considers: "If I take action A, I will be in state X. From state X, the remaining path to the goal looks like Y." This is the basis for Plan-and-Execute, Monte Carlo Tree Search-based agents, and multi-step lookahead planners.

Reactive systems are simpler and lower-latency. They work well when tasks are short, actions are reversible, and errors are easy to recover from. Deliberative systems are better for long-horizon tasks where early mistakes are costly and hard to undo (e.g., a financial trade execution agent, a multi-step deployment pipeline).

**Tree-of-Thoughts (ToT)** is a deliberative technique where the agent maintains a tree of partial reasoning paths instead of one linear chain: at each step it generates several candidate "thoughts," evaluates them, and expands the most promising while pruning dead ends — allowing it to backtrack to an earlier branch point rather than being stuck with a bad path. It maps directly onto classical tree search (BFS, DFS, MCTS) applied to LLM reasoning.

In practice, ToT multiplies model calls: generating and evaluating multiple candidates per step is expensive and slow. It is most justified for tasks that are inherently search-like — combinatorial planning, constrained code synthesis, or puzzles where many paths must be explored before the right one emerges. For most agentic tasks, ReAct or Plan-and-Execute is more economical. The conceptual contribution of ToT is establishing that single-path (chain) reasoning is a special case, and that deliberately exploring multiple paths is sometimes worth the cost.

## 2.7 Comparison Table: Reasoning Patterns

| Pattern | Also Known As | Core Mechanism | Planning Horizon | Self-Correction | Best For | Key Weakness |
|---|---|---|---|---|---|---|
| **ReAct** | Reason-Act, Thought-Action-Observation | Interleaved thought + tool call | Single step | None (by default) | Short tasks, transparent reasoning | No lookahead; degrades on long tasks |
| **Plan-and-Execute** | Hierarchical planner, decompose-and-act | Upfront plan, then execute | Multi-step | Replanning on failure | Long-horizon tasks with stable structure | Plan errors amplify; bad at adaptive tasks |
| **Reflexion** | Self-critique, iterative refinement | Self-evaluation after each attempt | Per-attempt | Yes, via stored reflection | Tasks with verifiable outcomes | Needs reliable self-diagnosis |
| **Chain-of-Thought** | CoT, scratchpad reasoning | Step-by-step text reasoning | Within a single call | No (one pass) | Math, logic, decomposition | No external state; reasoning can still err |
| **Tool-Augmented CoT** | ReAct variant, augmented reasoning | CoT where some steps call tools | Single step per tool | No | Mixed reasoning + data-lookup tasks | Tool call latency; schema complexity |
| **Deliberative / Lookahead** | MCTS agent, tree-of-thought | Search over action sequences | Multi-step lookahead | Implicit (search prunes bad paths) | High-stakes, irreversible actions | Computationally expensive |

## 2.8 Test Yourself

**Q2.1** Explain ReAct to someone who has never heard of it. What problem does it solve compared to a model that just generates an answer directly?

**Q2.2** You are building an agent that needs to book multi-leg international flights — including visa checks, connection time validation, and price comparison across three APIs. Which planning pattern would you choose, and why?

**Q2.3** What is the practical difference between Reflexion and a simple retry loop? Give a scenario where Reflexion meaningfully outperforms simple retry.

**Q2.4** A model is asked to compute compound interest over 20 years. It produces a slightly wrong answer using chain-of-thought. Would adding a calculator tool fix the problem? What would fix it and what wouldn't?

**Q2.5** You have a ReAct agent completing a 30-step research task. At step 18, the agent's reasoning becomes repetitive and it starts calling the same tool with the same query. What is happening, and how would you fix it architecturally? (Loop detection recurs deliberately in this book — see also Q9.5 and the debugging decision tree in QR-6, each from a different angle: reasoning pattern, failure taxonomy, and production monitoring.)

**Q2.6** Compare reactive and deliberative planning for a software deployment agent that runs database migrations. Which is safer and why?

**Q2.7** A team proposes using Plan-and-Execute for a customer service bot that handles open-ended inquiries. You have concerns. What are they?

**Q2.8** When would you *not* use chain-of-thought prompting? Give a complete answer.

**Q2.9** Design a Reflexion loop for a code-writing agent. What constitutes a valid reflection? What signal would you use to trigger it?

**Q2.10** Describe a scenario where ReAct and Plan-and-Execute would produce meaningfully different outcomes for the same task. Walk through both approaches.

**Q2.11** What is Tree-of-Thoughts and how does it differ from chain-of-thought and from ReAct? Describe a task type where ToT is clearly superior to a linear reasoning chain, and one where it is not worth the cost.

**Q2.12** Walk through the pseudocode for a ReAct agent loop. At each step, identify what the model does, what the harness does, and what could go wrong.

## Further reading

- Chen, Y., Benton, J., et al. (2025). "Reasoning Models Don't Always Say What They Think". [https://arxiv.org/abs/2505.05410](https://arxiv.org/abs/2505.05410)
- Shinn, N., Cassano, F., et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning". https://arxiv.org/abs/2303.11366
- Wei, J., Wang, X., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". https://arxiv.org/abs/2201.11903
- Yao, S., Zhao, J., et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models". https://arxiv.org/abs/2210.03629
- Yao, S., Yu, D., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models". https://arxiv.org/abs/2305.10601
- Zhou, A., Yan, K., et al. (2023). "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models". https://arxiv.org/abs/2310.04406
