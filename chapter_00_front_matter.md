---
pdf_options:
  format: Letter
  printBackground: true
  displayHeaderFooter: true
  headerTemplate: |-
    <span></span>
  footerTemplate: |-
    <section style="font-size: 9px; width: 100%; text-align: center; color: #888;">
      <span class="pageNumber"></span>
    </section>
---

# Architecting AI Agents: Building, Evaluating, and Shipping Agentic AI Systems

*A practical guide for current and aspiring AI architects, with a full interview-prep companion*

---

## Table of Contents

1. Foundations
2. Core Reasoning & Planning Patterns
3. Tool Use & Harness Engineering
4. Memory Architectures
5. Retrieval-Augmented Generation for Agents
6. Multi-Agent Systems
7. Frameworks and Protocols
8. Evaluation of Agentic Systems
9. Failure Modes and Guardrails
10. Security, Safety, and Governance
11. Production Operations and Reliability
12. Model and Architecture Fundamentals (Agent-Relevant Subset)
13. Platform Thinking and Scale
14. Interview Appendix A: System Design Interview Rounds
15. Interview Appendix B: Leadership, Influence, and Behavioral Questions
16. Answers Appendix
17. Quick Reference
18. Glossary

---

## What We Assume You Already Know

This book is for engineers and researchers building, evaluating, and shipping agentic AI systems. Chapters 1-13 are the core guide, and Interview Appendices A-B are a companion for using this material to prepare for an agentic AI interview. Either way, this is not a from-scratch introduction to machine learning. To get full value, you should already be comfortable with the following.

- **Transformers and attention:** what self-attention computes, why context length matters, and roughly what a KV cache is (Chapter 12 covers the agent-relevant subset).
- **LLM inference basics:** tokens, prompts, sampling/temperature, and the difference between a base model and an instruction-tuned/chat model.
- **Embeddings and vector similarity:** what an embedding is and why nearest-neighbor search over embeddings powers retrieval.
- **Prompting fundamentals:** system vs. user messages, few-shot examples, and structured (JSON) output.
- **General software engineering:** APIs, JSON, async/concurrency at a conceptual level, and reading a stack trace or log.
- **Basic ML training vocabulary:** supervised vs. reinforcement learning, fine-tuning, and preference data (needed for the alignment material in Chapter 10).
