# Interview Appendix A: System Design Interview Rounds

These two appendices apply the core chapters for interview prep. This section covers system design rounds, and Appendix B that follows is about behavioral and leadership rounds.

## 14.1 What the Interviewer Is Assessing

System design rounds for agentic AI roles probe:
- **Depth of understanding:** Can you describe the actual components, not just wave your hands at "an agent"?
- **Tradeoff reasoning:** Can you articulate what you give up with each design choice?
- **Production awareness:** Do you think about failure, latency, cost, and observability, not just task completion?
- **Scoping judgment:** Can you ask clarifying questions that reveal what actually matters in the design?

The format: you are given a vague problem statement ("design an agent for X"), you ask clarifying questions, then you whiteboard an architecture. The interviewer may challenge your choices, add constraints, or ask you to handle failure cases.

## 14.2 How to Approach a System Design Question

**Step 1 — Clarify requirements (3-5 minutes):**
- What does success look like? (Task completion rate, latency, cost?)
- Who are the users? (Internal, external, how many?)
- What tools/data sources must the agent use?
- What are the failure consequences? (High-stakes? Reversible?)
- What are the scale requirements? (Requests per day, concurrent sessions?)
- Are there compliance or data residency requirements?

**Step 2 — Sketch the happy path (5-7 minutes):**
Draw the core agent loop for the simplest version of the task: input, reasoning, tool calls, output. Name each component.

**Step 3 — Identify the hard parts:**
Where will this design break? What failure modes are specific to this task? What aspects of the design are uncertain?

**Step 4 — Address failure and production concerns:**
How does the system recover from tool failures? What observability does it have? How is cost bounded?

**Step 5 — Discuss tradeoffs:**
Every choice you made has an alternative. Be prepared to defend your choices and articulate what you would choose differently under different constraints.

## 14.3 Worked Example: Long-Horizon Research Agent

**Problem:** Design an agent that can research a given company and produce a 10-page due-diligence report, starting from just the company name.

**Clarifying questions:**
- How long can the agent run? (Minutes? Hours?)
- Are there sources it must consult (SEC filings, news, LinkedIn)?
- Who reviews the output? (Analyst, investor?)
- What is the budget per report?

**Architecture sketch:**

```
User input: "Company: Acme Corp"
                    │
         ┌──────────▼───────────┐
         │   Orchestrator       │
         │   (Planner)          │
         │   - Generates plan   │
         │   - Tracks progress  │
         └──────────┬───────────┘
                    │ delegates parallel subtasks
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
 [Researcher A]  [Researcher B] [Researcher C]
 (SEC filings)   (News/press)   (LinkedIn/team)
     │              │              │
     └──────────────┼──────────────┘
                    ▼
         ┌──────────────────────┐
         │   Synthesis Agent    │
         │   - Deduplicates     │
         │   - Structures       │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │   Critic Agent       │
         │   - Checks for gaps  │
         │   - Flags uncertain  │
         └──────────┬───────────┘
                    ▼
                 Final report
```

**Key design decisions:**
- Parallel research agents reduce latency; each has its own context window
- Separate synthesis and critic agents enforce separation of concerns
- Orchestrator checkpoints state so the task can resume if interrupted
- Budget cap on total tokens per report
- Human review of final report before delivery (due-diligence has high stakes)

**Failure handling:**
- If a research agent fails (tool timeout), the orchestrator retries once, then marks that section as "data unavailable" and continues
- If the synthesis agent produces a report under a minimum length threshold, the orchestrator re-invokes the researcher with a gap-fill query

## 14.4 Worked Example: Tool Failure and Timeout Handling

**Problem:** Your agent depends on an external CRM API that is flaky (5% error rate, occasional 30-second timeouts). Design the tool layer to handle this gracefully.

**Design elements:**
1. **Retry with exponential backoff:** Retry tool calls on transient errors (5xx, timeout) with 1s, 2s, 4s delays; max 3 retries.
2. **Circuit breaker:** After N consecutive failures, open the circuit: stop calling the API and return a "service unavailable" result immediately. Reset after a cool-down period.
3. **Partial result handling:** Design the tool response schema to include a `confidence` field and a `completeness` field. The agent can reason about incomplete results rather than treating partial responses as complete.
4. **Fallback data source:** If the CRM is unavailable, fall back to a cached copy (potentially stale, but marked as such).
5. **Graceful degradation:** The agent should be able to complete the task with reduced capability when the CRM is unavailable. Design the task completion logic to handle the "CRM unavailable" state explicitly.

## 14.5 Choosing Between Complexity and Simplicity

**Problem pattern:** A multi-agent architecture achieves 2% higher benchmark scores than a single agent but is 4x more complex and 3x more expensive. What do you build?

**The right frame:** Benchmark improvement is not the same as production value. Ask:
- Does 2% matter for this use case? (Medical diagnosis: yes. Email drafting: probably not.)
- What is the maintenance cost of the complex system over time?
- Does the complex system degrade gracefully, or does it fail catastrophically?
- What is the opportunity cost? (Could the engineering time spent on multi-agent be spent on better evals or better tooling?)

**The default:** Start with the simpler system. Measure its performance in production. Add complexity only when you have clear evidence that a specific capability is the bottleneck. "We might need multi-agent later" is not sufficient justification for multi-agent now.

## 14.6 Worked Example: Diagnosing a Broken Trace

System-design rounds increasingly hand you a *failing* trace and ask you to find the fault, not just design a happy path. This mirrors real on-call work and the debugging decision tree in QR-6. Below is an annotated trajectory for a research agent asked: *"What was Acme Corp's total 2024 revenue, and how did it compare to 2023?"* The final answer is confidently wrong. Walk it the way you would at a whiteboard.

```
step 1  REASON  "I need Acme's 2024 and 2023 revenue. I'll search the filings store."
step 2  ACT     search_filings(query="Acme Corp revenue")          ← no year in query
step 3  OBS     4 chunks returned, top score 0.74:
                 #1 "Acme Corp FY2022 revenue was $4.1B..."         (0.74)
                 #2 "Acme Corp revenue grew year over year..."      (0.72)
                 #3 "...Acme's 2023 revenue of $5.2B..."            (0.71)
                 #4 "Acme Holdings (unrelated) 2024 revenue $9B"    (0.70)
step 4  REASON  "The top result gives revenue of $4.1B. I have what I need."
step 5  ACT     finish(answer="Acme's 2024 revenue was $4.1B, up from 2023.")
```

**Diagnosis, step by step:**

1. **The query is under-specified (step 2).** The agent searched `"Acme Corp revenue"` with no year filter and no metadata constraint. Retrieval can only rank on the query it was given; the years the task actually needs (2024, 2023) never entered the query. This is the root cause — everything downstream inherits it.
2. **Retrieval silently returned near-misses (step 3).** All four chunks score in a tight 0.70-0.74 band, which should be a yellow flag: no chunk clearly wins, and the scores do not distinguish the relevant year from irrelevant ones. Worse, chunk #4 is a *different company* (Acme Holdings vs. Acme Corp) that happens to mention "2024 revenue" — an entity-collision failure the agent never checks for.
3. **The agent trusted rank-1 without checking the field it needed (step 4).** It grabbed the top-scored chunk ($4.1B) even though that chunk is FY2022 — neither year the task asked about. The relevant 2023 figure ($5.2B) was sitting at rank 3 and was ignored; no 2024 figure for Acme *Corp* was retrieved at all. The reasoning step performed no verification that the retrieved value matched the requested year or entity.
4. **It fabricated the comparison (step 5).** "up from 2023" is asserted with no 2023 figure actually used and no 2024 figure retrieved. This is a grounding failure layered on top of the retrieval failure.

**Where it silently failed:** the failure is at step 2/3 (retrieval), but it was *invisible* because the pipeline had no verification gate between "retrieved a chunk" and "used the chunk." The final answer looks fluent, so a final-answer eval would likely pass it — exactly why trajectory-level inspection matters (see 8.3).

**The fixes, mapped to the failure points:**
- **Query construction:** decompose the task and issue year-scoped queries (`Acme Corp 2024 revenue`, `Acme Corp 2023 revenue`) or apply metadata filters (`entity=Acme Corp`, `year in {2023,2024}`) before the vector search (see 5.5 query transformation, 5.7 metadata filtering).
- **Retrieval-quality signal:** treat a flat score band and low absolute scores as "low-confidence retrieval," and escalate (re-query, widen, or ask for clarification) rather than proceeding.
- **Entity/field verification:** before using a chunk, check that its entity and year match what the task asked for — a cheap self-check step that would have rejected chunk #1 (wrong year) and chunk #4 (wrong entity).
- **Grounding gate:** require every figure in the final answer to be traceable to a retrieved chunk with the matching entity and year; refuse or flag when the comparison year is missing rather than inventing a direction.

Saying "the retrieval was bad" is not enough in an interview. The signal an interviewer wants is that you can name *where* the pipeline had no check, *why* the failure stayed silent, and *which specific gate* you would add at each point.

## 14.7 Whiteboard Architecture Components

A well-architected agentic system has these components at the whiteboard level:

```
┌─────────────────────────────────────────────────────────────┐
│                         API Gateway                         │
│         (Auth, rate limiting, request routing)              │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                Agent Orchestration Layer                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Planner /  │  │  State &     │  │  Memory Manager    │  │
│  │  Reasoner   │  │  Checkpoint  │  │  (context window   │  │
│  │  (LLM call) │  │  Store       │  │   + external mem)  │  │
│  └──────┬──────┘  └──────────────┘  └────────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │ tool calls
┌─────────▼───────────────────────────────────────────────────┐
│                      Tool Execution Layer                   │
│  ┌────────────┐  ┌────────────┐  ┌───────────────────────┐  │
│  │  Tool      │  │  Tool      │  │  Sandbox / Security   │  │
│  │  Registry  │  │  Dispatcher│  │  Layer                │  │
│  └────────────┘  └────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │ responses
┌─────────▼───────────────────────────────────────────────────┐
│                   Observability & Safety Layer              │
│    Tracing  │  Guardrails  │  Audit Log  │  Cost Monitor    │
└─────────────────────────────────────────────────────────────┘
```

Know what each layer does, what can go wrong in it, and how you would monitor it.

## 14.8 Test Yourself

**Q14.1** Design an agent that monitors a company's Slack and Jira and sends a weekly summary email to leadership. Walk through the architecture, the tools required, the memory design, and the failure handling.

**Q14.2** You are asked to design a coding agent that can implement a feature from a GitHub issue, run tests, and open a pull request. What are the most important safety guardrails, and where in the architecture do they sit?

**Q14.3** An interviewer gives you this constraint: "The agent must never spend more than $2 on a single task." How would you implement this? Where in the architecture does the check live?

**Q14.4** Design the retry and fallback strategy for an agent that depends on three external APIs (CRM, billing system, inventory database). How do you handle the case where two of the three are unavailable?

**Q14.5** Compare two architectures for an email triage agent: (A) a single agent with all tools, and (B) a router agent that dispatches to specialist subagents by email type. Under what conditions does B outperform A, and when is A sufficient?

**Q14.6** Sketch the whiteboard architecture for a multi-tenant agent platform serving five enterprise customers. Identify which components are shared and which are per-tenant.

**Q14.7** You are building an agent that can modify production database records. The interviewer says, "How do you make sure this doesn't accidentally delete everything?" Walk through your complete safety architecture.

**Q14.8** An interviewer asks: "Why not just use a single very large context window with all available information instead of building an agent with retrieval?" Give a complete answer.

**Q14.9** Design an agent that can answer questions about a 10,000-document internal knowledge base. The P99 latency requirement is under 5 seconds. Walk through every component and the latency budget.

**Q14.10** You are designing an agent for a medical information platform. A user asks the agent for medication dosage advice. Walk through the safety architecture specific to this scenario.
