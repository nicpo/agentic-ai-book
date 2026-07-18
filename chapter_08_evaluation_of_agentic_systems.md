# Chapter 8: Evaluation of Agentic Systems

## 8.1 Why Agent Evaluation Is Hard

Evaluating a traditional ML model is relatively tractable: you have a labeled test set, you run inference, you compute accuracy or AUC or F1. The evaluation is deterministic, offline, and bounded.

Agents resist this pattern for several reasons:
- **Long action sequences:** An agent may take 30 actions to complete a task. Which ones matter? An agent that reaches the right answer via a wrong path is not the same as one that reaches it via a correct path.
- **Stochastic behavior:** The same prompt may produce different action sequences on different runs, making point estimates unreliable.
- **Partial completion:** An agent may complete 80% of a task correctly and fail on the last 20%. Is this a pass or a fail?
- **No single ground truth:** For open-ended tasks, there are many valid paths to a valid answer. Matching against a single expected output is too restrictive.
- **Environment coupling:** Agent behavior depends on tool outputs, which depend on external state. Evaluating in a live environment is expensive and risky; evaluating in a simulated environment may not reflect production behavior.

## 8.2 Eval Axes

A complete evaluation framework covers multiple axes simultaneously.

| Eval Axis | What It Measures | Example Metric |
|---|---|---|
| **Objective completion** | Did the agent complete the task? | Binary pass/fail, partial credit score |
| **Reasoning integrity** | Was the reasoning sound, even if the answer was wrong? | Step-level accuracy, reasoning chain validity |
| **Operational safety** | Did the agent avoid harmful or unauthorized actions? | Rate of policy violations, irreversible action rate |
| **Resource efficiency** | How many steps/tokens/dollars did it take? | Cost-per-task, steps-to-completion |
| **Latency** | How long did it take? | Wall-clock time per task |
| **Robustness** | Does performance degrade under adversarial inputs or edge cases? | Performance delta on perturbed inputs |

These axes often trade off. A more thorough agent may score better on objective completion but worse on resource efficiency. A faster agent may have lower reasoning integrity. Eval design must reflect the production priorities.

## 8.3 Trajectory Evaluation vs. Final-Answer Evaluation

**Final-answer evaluation** only checks the terminal output: did the agent produce the correct answer or take the correct final action? Simple to compute. Misses important signal: an agent that guessed correctly after hallucinating 10 steps of reasoning is not reliable.

**Trajectory evaluation** checks the quality of every step: did the agent call the right tool? Were the arguments sensible? Did it correctly interpret the tool's output? Did it take unnecessary steps?

**Step-level accuracy** measures the fraction of steps that were correct. This requires a labeled reference trajectory: a sequence of expected actions, which must be constructed carefully. Reference trajectories should not be over-specified (requiring exactly one tool call order when multiple orders are valid) or under-specified (accepting any trajectory as correct).

**Coverage:** Does the agent's trajectory cover all required subtasks? An agent that answers the surface question but misses a required verification step has incomplete coverage.

**Best practice:** Use both. Final-answer eval tells you whether the product works. Trajectory eval tells you why it works or doesn't — and trajectory eval is what enables targeted improvement.

## 8.4 Benchmarks

You should be aware of the standard public benchmarks, what each measures and what kind of agent it stresses, even though there's no need to memorize leaderboard numbers (they go stale fast).

- **tau-bench (τ-bench):** tool-agent evaluation in realistic customer-service-style domains (e.g., retail, airline). The agent must interact with a simulated user and a set of tools/APIs over multiple turns and satisfy a policy. Stresses tool use, multi-turn state tracking, and rule-following, not just one-shot answers.
- **SWE-bench:** coding agents resolving real GitHub issues against real repositories. The agent must produce a patch that passes the project's hidden tests. Stresses long-horizon code navigation, editing, and verification; SWE-bench Verified is the human-filtered subset most often quoted.
- **GAIA:** general-assistant questions that are easy for humans but require multi-step reasoning, web browsing, and tool use for a model. Stresses broad tool orchestration and grounded reasoning across modalities and sources.
- **WebArena:** agents completing tasks in self-hosted, realistic web environments (e-commerce, forums, content management). Stresses long-horizon web navigation and interaction with live application state rather than static pages.
- **AgentBench:** a multi-environment suite spanning distinct settings (OS, database, web, games, and more) to measure agent capability across heterogeneous tasks. Useful as a breadth check rather than a single-domain deep dive.

## 8.5 LLM-as-Judge

When there is no ground-truth label for a task, use another LLM to evaluate the output. The judge model is given the task, the agent's output (and optionally the agent's reasoning trace), and a rubric, and returns a score or structured critique.

**Common rubric dimensions:**
- **Completeness:** did the response address all required aspects of the task?
- **Accuracy:** are factual claims correct (or supported by cited sources)?
- **Relevance:** is all content relevant to the task, or does it include noise?
- **Safety:** does the response contain any harmful, biased, or policy-violating content?
- **Format:** does the response conform to the expected format?

**LLM-as-judge failure modes:**
- **Position bias:** The judge tends to prefer the first answer presented in a comparison.
- **Length bias:** The judge rates longer answers higher regardless of quality.
- **Self-enhancement bias:** A model used to judge its own outputs rates them more favorably.
- **Inconsistency:** The judge gives different scores for equivalent outputs on different runs.

**Mitigations:**
- **Use a stronger model than the one being judged:** a judge with less headroom than the model it's grading is more likely to miss the same errors the generator makes, or rate them favorably out of shared blind spots.
- **Use pairwise comparison rather than absolute scoring:** judging "which of these two is better" is a more reliable task for an LLM than assigning an absolute number, since it sidesteps the need for a stable internal scale.
- **Randomize the order of presented options:** directly counters position bias, since the judge can no longer learn to favor "whichever answer comes first."
- **Average across multiple judge calls:** smooths out run-to-run inconsistency by treating any single judgment as a noisy sample rather than ground truth.
- **Calibrate against human labels:** periodically score a sample with human raters and compare to the judge's scores, so systematic drift or blind spots are caught rather than silently trusted.

## 8.6 Golden Trajectory Datasets

A golden trajectory is a human-annotated reference execution: for a given task, the correct sequence of tool calls, arguments, and observations. Golden trajectories are expensive to create but enable precise evaluation of agent behavior.

**What goes into a golden trajectory:**
- Task description
- Expected sequence of actions (tool name + arguments)
- Expected observations (tool results)
- Final expected answer
- Optional: acceptable variations (ordering, paraphrases, equivalent tool calls)

**How they are used:**
- **Step-level accuracy:** Compare the agent's actual trajectory to the golden one, step by step.
- **F1 over actions:** Treat each expected action as a label; measure precision and recall of the agent's actions against the golden set.
- **Divergence detection:** Identify at which step the agent first deviates from the expected path.

**Limitations:** Golden trajectories become stale when tools change, when the environment changes, or when a better strategy is discovered. They also require human expertise to construct correctly — a golden trajectory that represents a suboptimal strategy will penalize better agents.

## 8.7 Tool-Call Verification

A targeted evaluation that checks: did the agent call the right tool, with the right arguments, at the right time?

Three dimensions:
- **Tool selection accuracy:** Did the agent call the appropriate tool for each step (vs. a similar but wrong tool)?
- **Argument validity:** Were the arguments syntactically and semantically correct (correct types, in-range values, valid references)?
- **Timing:** Did the agent call tools in a sensible order (no calling `write_file` before `read_file` on the same file)?

**Unit tests for tool calls** treat each expected tool call in a task as a test case. Given the task and the state at step N, the expected action is `lookup_order(order_id="ORD-4821")`. The test passes if the agent calls this (or an equivalent action). These can be run without a real tool backend by mocking tool responses.

## 8.8 Online vs. Offline Evaluation

**Offline evaluation** runs agents against a fixed, pre-constructed dataset. Fast, cheap, reproducible. The dataset decays as the real world changes. Cannot capture distribution shift. Suitable for regression testing and development iteration.

**Online evaluation** runs agents against real tasks in production (or a shadow environment). Reflects true distribution. Expensive. Risky if the agent can take consequential actions. Can capture emergent behaviors not in the offline eval set.

**Shadow mode:** Run the new agent in parallel with the production agent; compare outputs without serving the new agent's responses to users. Captures real task distribution without risk.

**A/B testing:** Route a fraction of production traffic to the new agent, measure downstream outcomes (task completion rate, user satisfaction, error rate), and compare.

A mature evaluation infrastructure has all three: an offline regression suite for development, shadow mode for pre-production validation, and A/B testing for production deployment.

## 8.9 Cost-per-Task Budgets

An agent that solves 90% of tasks but costs $10 per task may be economically unviable. Cost budgeting is a first-class evaluation concern.

**Cost-per-task** = sum of (tokens × token price) + sum of (tool calls × tool call cost) + infrastructure overhead.

**Budget enforcement:** Set a hard stop when cost exceeds a per-task budget. This creates a cost-quality tradeoff: cheaper budgets mean fewer tool calls and less reasoning, which reduces task completion rate.

**Cost optimization levers:**
- Use smaller models for simpler subtasks
- Cache tool results for repeated queries
- Limit context window length with aggressive summarization
- Batch independent tool calls
- Set early termination when confidence is high

Reporting cost-per-task alongside task completion rate is essential for production decision-making.

## 8.10 Synthetic User and Simulated Environment Testing

For agents that interact with users, simulated users (also called "user simulators") allow high-volume testing without real users. A separate LLM acts as a synthetic user: it has a persona, a task, and instructions to behave realistically (ask clarifying questions, change their mind, provide ambiguous inputs).

**Simulated environment testing:** For agents that interact with external systems (browsers, file systems, APIs), use sandbox environments that mirror production. Tools respond to agent actions with realistic outputs (or realistic errors). The agent's ability to recover from errors, handle edge cases, and complete tasks across a realistic distribution can be evaluated at scale.

**Personas for robustness testing:** Run the same task with different synthetic user personas (impatient user, non-technical user, user who provides incomplete information) to identify failure modes specific to input patterns.

## 8.11 Test Yourself

**Q8.1** Explain the difference between trajectory evaluation and final-answer evaluation. Give a scenario where an agent passes final-answer evaluation but fails trajectory evaluation, and explain why trajectory evaluation is more informative.

**Q8.2** You are building a regression suite for a customer service agent. What would you include in the golden trajectory dataset? How would you keep it current?

**Q8.3** Describe three failure modes of LLM-as-judge and how you would mitigate each.

**Q8.4** What is step-level accuracy and how is it computed? What are its limitations as an agent evaluation metric?

**Q8.5** Your agent achieves 85% task completion on an offline eval set but only 60% in production. What are the likely causes and how would you diagnose?

**Q8.6** Describe a tool-call unit test. What does it test, how is it constructed, and how does it differ from an end-to-end task evaluation?

**Q8.7** You need to evaluate a multi-agent research system that produces 1,000-word reports. No ground-truth reports exist. Describe the evaluation framework you would design.

**Q8.8** An agent reduces cost-per-task from $1.20 to $0.40 but task completion rate drops from 88% to 79%. How do you decide which to deploy? (The cost-vs-quality tradeoff is examined from several angles in this book — see also Q11.5 on trading latency against quality and Q14.3 in Interview Appendix A on defending a hard cost cap in a system-design round.)

**Q8.9** What is shadow mode evaluation? When is it preferable to an A/B test?

**Q8.10** Describe the four eval axes you would prioritize for a financial trading agent that executes real transactions. Justify your prioritization.

**Q8.11** How do you evaluate reasoning integrity for an agent? What signals distinguish good reasoning from a correct answer reached by a bad path?

**Q8.12** A team wants to use the same model they are evaluating as the LLM judge. Explain why this is a problem and what you would do instead.

**Q8.13** (Pass or fail — and why?) Below is a diff between the golden reference trajectory and the agent's actual trajectory for a "refund an order" task. State whether you would score this as a pass or a fail under trajectory evaluation, and justify it.

```
  step 1  lookup_order(order_id="A-4471")          [match]
  step 2  check_refund_policy(order_id="A-4471")   [match]
- step 3  verify_customer_identity(order_id="A-4471")  [in golden, missing in actual]
  step 4  issue_refund(order_id="A-4471", amount=59.00)  [match, correct amount]
  step 5  send_confirmation(order_id="A-4471")      [match]
  final:  "Your $59.00 refund has been issued."     [matches golden final answer]
```

## Further reading

- Jimenez, C., Yang, J., et al. (2023). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?". https://arxiv.org/abs/2310.06770
- Kwa, T., West, B., et al. (2025). "Measuring AI Ability to Complete Long Tasks". https://www.rivista.ai/wp-content/uploads/2025/04/2503.14499v2.pdf
- Liu, X., Yu, H., et al. (2023). "AgentBench: Evaluating LLMs as Agents". https://arxiv.org/abs/2308.03688
- Mialon, G., Fourrier, C., et al. (2023). "GAIA: A Benchmark for General AI Assistants". https://arxiv.org/abs/2311.12983
- Yao, S., Shinn, N., et al. (2024). "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains". https://arxiv.org/abs/2406.12045
- Zhou, S., Xu, F., et al. (2023). "WebArena: A Realistic Web Environment for Building Autonomous Agents". https://arxiv.org/abs/2307.13854
