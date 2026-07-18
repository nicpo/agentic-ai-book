# Chapter 9: Failure Modes and Guardrails

## 9.1 Taxonomy of Agent Failure Modes

Agents fail in ways that differ structurally from traditional software failures and from ML model failures. The failure modes are important to know for designing robust systems.

**Hallucinated tool calls:** The agent generates a tool call to a function that does not exist, with arguments that do not match the actual schema, or with a plausible-looking but incorrect argument value (e.g., a customer ID that doesn't exist). The harness receives an invalid call, returns an error, and the agent must recover.

**Infinite loops:** The agent repeatedly calls the same tool with the same arguments, receives the same result, and fails to make progress. Common cause: the agent's reasoning fails to update given repeated identical observations. The agent concludes it needs more information, gets the same information, and concludes again that it needs more.

**Partial execution:** The agent completes some steps of a multi-step task but fails mid-way, leaving the system in an inconsistent state. Example: an agent that books a flight but fails before confirming the hotel — the user now has a flight with no hotel and the agent has marked the task as incomplete.

**Incorrect actions:** The agent takes an action that is valid syntactically and executable, but wrong for the task. Example: deleting a file the user meant to archive, or sending an email draft instead of saving it.

**Context window overflow:** The accumulated history of actions, observations, and reasoning exceeds the model's context limit. If not handled, the model either truncates (losing early context) or fails.

**Cascading errors in multi-agent chains:** An error in an upstream agent produces incorrect output that a downstream agent treats as ground truth, amplifying the error. Discussed in detail in Chapter 6.

**Tool timeout and partial results:** A tool call times out or returns partial results. If the agent treats a timeout as a successful empty result, it proceeds with incorrect information.

**Sycophantic drift:** In multi-turn interactions with users, the agent's responses gradually drift toward agreeing with the user's framing rather than maintaining accurate, grounded reasoning.

## 9.2 Human-in-the-Loop (HITL) Patterns

Human oversight is the most reliable guardrail for high-stakes agentic actions. The design space is how much oversight, at what points, with what granularity.

**Approval gates:** Pause execution before irreversible or high-risk actions and require explicit human approval. The implementation in LangGraph: a conditional node that routes to a "waiting for approval" state; execution resumes only when the approval signal is received.

**Threshold-based gating:** Automate low-risk actions; require approval for actions above a risk threshold. Example: a financial agent executes trades under $10,000 autonomously; trades over $10,000 require human approval.

**Review-and-submit:** The agent drafts a complete plan or output and presents it to the human before any action is taken. The human reviews and approves the full plan, not individual steps. Lower friction for the human than step-by-step approval; appropriate when the plan can be fully specified in advance.

**Audit-only:** The agent acts autonomously but every action is logged with full detail. Human reviews the log asynchronously. Appropriate when actions are low-risk but auditability is required for compliance.

**Escalation paths:** Define conditions under which the agent should explicitly request human input: ambiguous instructions, conflicting information, task outside the defined scope, or confidence below a threshold.

## 9.3 Guardrails: Input and Output

**Input classification:** Before passing user input to the agent, classify it for intent and safety. Route adversarial inputs, off-topic requests, and policy-violating content away from the main agent before it can act on them.

**Output filtering:** After the agent generates a response or an action, check it against policy before delivering it. Filter PII in outgoing messages, check responses for policy violations, validate action arguments before execution.

**Rate limiting on dangerous tools:** Some tools should have hard limits on call frequency or arguments, regardless of what the agent requests. A `send_email` tool should not be callable 1,000 times per hour by an agent in a loop. A `delete_file` tool should not accept wildcard paths.

**Tool permissions by role:** Agents should have access only to the tools their role requires. A research agent does not need file-write access. A summarization agent does not need network access.

**Kill switches:** A manual or automated mechanism to halt an agent's execution immediately. The agent loop should check a "halt" flag before each action, and the flag should be settable by an external operator without waiting for the current action to complete.

**Runaway loop detection:** Monitor for patterns that indicate a stuck loop: identical tool calls repeated N times, increasing elapsed time without progress signals, cost-per-step rising without approaching task completion. Automatically halt and alert.

## 9.4 Guardrails: Infrastructure-Level

**Sandboxing:** Run agent-executed code in an isolated environment (container, VM, restricted subprocess). The agent should not be able to access the host filesystem, network, or other system resources beyond what is explicitly permitted.

**Network egress control:** Limit which external URLs the agent can contact. An agent that can make arbitrary HTTP requests can be manipulated into exfiltrating data or calling unintended services.

**Credential scoping:** Tools that use credentials (database connections, API keys) should use the minimum-permission credential for the task. An agent reading customer records should not have a credential that can also delete them.

**Execution time limits:** Every tool call and every agent iteration should have an enforced timeout. Without hard timeouts, a blocked agent can run indefinitely.

## 9.5 Handling Specific Failure Modes

**For infinite loops:** Maintain a loop-detection counter per tool-call pattern. If the same (tool, arguments) pair appears N times without a different observation, inject a "you appear to be stuck" message and suggest alternatives or halt.

**For partial execution:** Design tasks as transactions where possible: either all steps complete or none are committed. External API calls can't be made transactional this way because there is no shared transaction coordinator spanning your system and the external service — once the call reaches the external system, it has either committed there or it hasn't, and there is no atomic way to join that commit with your own. For these steps, maintain a compensation log instead: a record pairing each action taken with the operation that undoes it (e.g., `charge_card` paired with `refund_card`, `create_ticket` paired with `close_ticket`). If a later step in the sequence fails, the harness walks the log backward and executes the inverse of each completed action, approximating a rollback (the saga pattern).

**For context overflow:** Implement a context budget monitor. When approaching the limit, automatically summarize the oldest portion of the history and replace it with the summary. Log that summarization occurred so debugging can flag potential context loss.

**For hallucinated tool calls:** The harness should return a structured error message describing the valid tool set and the schema of the called tool, rather than a generic error. This gives the agent the information it needs to correct itself.

## 9.6 Test Yourself

**Q9.1** List five distinct agent failure modes and for each, describe one architectural mitigation.

**Q9.2** What is a "kill switch" in an agentic system and what are the requirements for it to be effective? Give an implementation sketch.

**Q9.3** Describe the approval gate pattern. For which categories of agent actions should approval gates be required by default?

**Q9.4** An agent calls `delete_files(path="/*")`. Describe the sequence of checks that should have caught this before execution, and identify where each check belongs in the system.

**Q9.5** Your agent is calling `search_web(query="quarterly earnings Apple")` on loop. Describe the loop detection mechanism you would implement and what happens when the loop is detected.

**Q9.6** What is partial execution failure? Give a concrete example and describe how you would design the system to recover from it.

**Q9.7** Explain the difference between input classification and output filtering as guardrail mechanisms. Give an example of a risk that input classification catches but output filtering would miss, and vice versa.

**Q9.8** Describe how you would implement rate limiting on a `send_email` tool used by an agent. What limits would you set, and how would you communicate limit violations back to the agent?

**Q9.9** A multi-step customer service agent is interacting with a user who insists that a clearly wrong policy interpretation is correct. The agent begins to agree with the user in subsequent turns. What failure mode is this and how would you prevent it architecturally?

**Q9.10** Design the guardrail stack for a code execution agent that can run arbitrary Python. List each guardrail layer, what it protects against, and where in the stack it sits.

**Q9.11** (Find the missing check.) This is the tool-dispatch path for a payments agent. A malicious or confused model can move money it should not. Identify the missing guardrail, name the failure class, and say exactly where the check belongs.

```python
TRANSFER_LIMIT = 10_000  # dollars, requires human approval above this

def dispatch(tool_call, ctx):
    fn = TOOLS[tool_call.name]
    if tool_call.name == "transfer_funds":
        return fn(
            from_acct=ctx.account_id,
            to_acct=tool_call.args["to_acct"],
            amount=tool_call.args["amount"],
        )
    return fn(**tool_call.args)
```

## Further reading

- OWASP. "Top 10 for Large Language Model Applications". https://genai.owasp.org/llm-top-10/
