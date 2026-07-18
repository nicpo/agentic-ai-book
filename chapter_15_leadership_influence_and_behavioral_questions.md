# Interview Appendix B: Leadership, Influence, and Behavioral Questions

## 15.1 What Senior Roles Require

Senior and staff-level agentic AI roles are evaluated not only on technical depth but on:
- **Technical direction:** The ability to make architectural decisions that other engineers follow.
- **Cross-team influence:** The ability to drive alignment without direct authority.
- **Judgment under uncertainty:** Making good bets when the research is not settled.
- **Production ownership:** The willingness to take responsibility for shipped systems, including when they fail.

Behavioral questions in these rounds are not an afterthought. They reveal whether a candidate has the judgment, the humility, and the persistence to be effective in a senior role — which a strong technical answer alone cannot demonstrate.

## 15.2 Leadership Stories: What Interviewers Are Looking For

> **A note on the example stories in this chapter.** The narratives below (and in 15.4) are *structural templates*, not scripts to memorize and recite. Use these to recognize what a strong story contains, then tell yours.

Good behavioral stories have:
- **Specificity:** Concrete situation, real stakes, named outcome. Not "I often try to..."
- **Your agency:** What you specifically decided, said, or did — not what the team did.
- **Non-obvious insight:** Something you saw or did that wasn't obvious to others. "I noticed that the evaluation metric was hiding a class of failure..."
- **Real consequence:** What happened as a result. "The team adopted the approach" or "We shipped on time" or "The system failed in production and here's what I learned."

**Common leadership story archetypes for agentic AI roles:**

**Setting a standard others adopted:** "I introduced trajectory evaluation to the team because our final-answer eval was missing a whole class of failures. I built a small internal tool and ran it alongside the existing eval for a month. When it caught three regressions the existing eval missed, the team adopted it as the primary eval framework."

**Course-correcting a project:** "We were three weeks from launch on a multi-agent system when I identified that the orchestrator was hallucinating task decompositions at a 12% rate on out-of-distribution inputs. The team was reluctant to delay. I presented data showing that 12% failure rate on our launch use case translated to roughly 2,400 failed tasks per day. We delayed two weeks and fixed it."

**Cross-team consensus:** "The security team wanted to block all external network calls from agents; the product team wanted unrestricted web access. I designed a proposal for tiered egress control — a whitelist for approved domains, with a review process for new domains — that both teams could accept. It took three meetings and two iterations of the proposal."

## 15.3 Agent Experimentation

Interviewers at research-adjacent companies may ask about your process for experimenting with agent designs. A strong answer describes:

- **Hypothesis-driven iteration:** "We hypothesized that adding a critic agent would improve output quality. We designed an A/B eval to test this before building the full integration."
- **Clear metrics upfront:** "Before any experiment, we defined what 'better' meant: step-level accuracy and cost-per-task, not just qualitative impressions."
- **Failure interpretation:** "The experiment failed to improve quality but revealed that the critic was surfacing a useful category of error we weren't tracking. We added that error category to our eval suite."

## 15.4 Failure Stories: "Tell Me About an Agent You Shipped That Failed"

This is a frequent, high-signal question for agentic AI roles. The interviewer wants to see:
- **Candor:** You will not pretend the failure didn't happen.
- **Root cause analysis:** You understood what went wrong technically.
- **System-level thinking:** You saw the systemic fix, not just the immediate patch.
- **Learning transfer:** What changed in your practice because of this failure?

**Example answer structure:**
"We shipped a document processing agent that performed well in offline eval but failed in production because our eval set was drawn entirely from one document format. Production had four additional formats we hadn't tested. The agent's chunking logic produced broken chunks on those formats, which caused the retrieval to fail silently — the agent proceeded with no retrieved context and hallucinated confidently. The fix was: format detection at ingestion, format-specific chunking logic, and coverage metrics in our eval set that tracked performance by document format. I now treat eval distribution coverage as a first-class shipping requirement."

## 15.5 Ethics and Autonomous Decision-Making

Senior interviews at companies deploying agents at scale will include ethics questions. Be prepared for:

**"Where would you refuse to let an agent act autonomously?"**

Strong answer: Identify the principles, not just examples.
- Decisions that affect people's lives or livelihoods (medical diagnosis, loan approvals, content moderation with real consequences) require human judgment in the loop.
- Irreversible actions (sending public communications, executing financial transactions, deleting data) require human review above certain thresholds.
- Actions taken outside the scope the user defined. An agent tasked with "summarize my emails" should not be sending emails, regardless of whether it believes this would help.
- Contexts where the agent cannot verify its own accuracy and the cost of error is high.

**"How do you handle bias in an agent that makes decisions affecting people?"**

Strong answer: Name specific bias types relevant to the system (selection bias in training data, confirmation bias in retrieval, representation bias in model outputs), describe how you would measure them (demographic disaggregation of eval metrics, adversarial test sets), and describe how you would address them (debiasing techniques, mandatory human review for affected demographic groups, bias monitoring in production).

## 15.6 Driving Standards Across Teams

**The question:** "Tell me about a time you drove a technical standard that was adopted beyond your immediate team."

What this probes: Senior engineers do not just write good code. They change what "good" looks like for the people around them.

**Elements of a strong answer:**
- The standard you identified as missing (e.g., no consistent approach to agent eval, or inconsistent tool schema conventions)
- How you diagnosed the impact (e.g., "each team was reinventing evaluation; results were not comparable across teams")
- How you drove adoption (wrote a proposal, ran a pilot, built tooling that made the right thing easy)
- The outcome and its scope (adopted by N teams, became an org-wide standard, led to a specific improvement)

## 15.7 Hiring and Team Building

Questions about hiring reveal how you think about the work. For agentic AI roles:

**"What do you look for in a candidate for an agentic AI engineer role?"**

Strong answer: technical fundamentals (systems thinking, debugging instinct, understanding of LLM behavior), product sense (what does success look like for a user?), comfort with ambiguity (the field is moving fast; you need people who experiment well, not just implement specifications), and intellectual honesty (the willingness to say "this didn't work and here's why" rather than finding reasons to declare success).

## 15.8 Test Yourself

**Q15.1** Tell me about a technical standard you introduced that was adopted by others. What was the standard, why did it matter, and how did you drive adoption?

**Q15.2** Describe a time you disagreed with a technical direction your team was pursuing. How did you raise the concern, and what happened?

**Q15.3** Tell me about an agent or AI system you shipped that failed in production. What happened, what was the root cause, and what changed in your practice?

**Q15.4** How do you approach experimenting with a new agent architecture when you don't know if it will work? Walk me through a specific example.

**Q15.5** Where would you refuse to allow an AI agent to act autonomously? What principles guide that judgment?

**Q15.6** You are the lead for an agentic AI team and you have identified that a competitor has an approach that significantly outperforms yours. Your team has been working on the current approach for eight months. How do you handle this?

**Q15.7** Describe a time you had to get cross-functional alignment (engineering, product, legal, security) on an AI system design. What was contested and how did you resolve it?

**Q15.8** How do you decide when an AI system is ready to ship? What are your criteria, and how do they differ from traditional software?

**Q15.9** A senior engineer on your team consistently over-engineers solutions, adding multi-agent complexity to problems that would be simpler with a single agent. How do you address this?

**Q15.10** Tell me about a time you had to explain a risk in an AI system to a non-technical stakeholder. What was the risk, how did you frame it, and what was the outcome?
