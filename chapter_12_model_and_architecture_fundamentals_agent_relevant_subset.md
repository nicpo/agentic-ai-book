# Chapter 12: Model and Architecture Fundamentals (Agent-Relevant Subset)

## 12.1 What You Need to Know

To create and debug agents, you need to know why context windows have limits, how attention scaling affects long-context performance, what alternative architectures offer, and how to choose between open and closed models for agentic deployment.

## 12.2 Transformer Self-Attention and Long-Context Coherence

The transformer architecture processes sequences by computing attention between every pair of tokens in the context. Each token attends to all other tokens, with attention weights determining how much each token influences the representation of every other.

**Why this matters for agents:** An agent accumulates a long context as it iterates — prompts, reasoning, tool calls, tool results. Transformers theoretically support any context length, but in practice:

- **Quadratic compute:** Standard attention computes O(n²) operations where n is the sequence length. A 100K-token context requires 10,000x more attention computation than a 1K-token context. This is why long-context models are slower and more expensive per call.
- **The lost-in-the-middle problem:** Models attend disproportionately to the start and end of the context, with worse recall for content in the middle. As of 2026 this effect in frontier models is model-dependent, as shown by long-context benchmarks (RULER, NoLiMa, multi-needle retrieval). For agents with long histories, this means early tool results and reasoning may not be well-attended, even if they are technically "in context."
- **Position encoding:** Transformers use position encodings to represent token position. Original absolute encodings degraded beyond training context lengths. Modern models use RoPE (Rotary Position Embedding) and other techniques that generalize better to longer sequences, but performance still degrades at very long contexts.

**Practical implication:** Don't assume that "it fits in the context window" means "the model will effectively use all of it." Active context management (summarization, selective retention) is often necessary even when context length is technically not exceeded.

## 12.3 Post-Transformer Architectures and SSMs

**State Space Models (SSMs)** and the **Mamba** architecture represent alternatives to the quadratic attention mechanism. Instead of attending to all previous tokens, SSMs maintain a fixed-size state that is updated as each new token is processed. This produces **linear** compute scaling with sequence length — O(n) instead of O(n²).

**What SSMs offer for agents:**
- Efficiently process extremely long sequences (millions of tokens) without quadratic cost
- Constant memory per step during inference (the state, not the full KV cache)
- Potentially better at tasks that require processing very long histories

**What SSMs sacrifice:**
- SSMs are theoretically less expressive than full attention for retrieval tasks (finding specific content from long ago in the sequence)
- In practice, Mamba and its successors have not yet matched frontier-tier transformer models on complex reasoning tasks *(as of mid-2026, the frontier-tier transformer models these are measured against are the flagship reasoning-tier offerings from Anthropic, OpenAI, and Google — check current benchmarks rather than assuming this gap has stayed the same size)*
- The ecosystem (tooling, fine-tuning, deployment infrastructure) is less mature

**Hybrid architectures** (Mamba layers + attention layers at specific positions) attempt to combine the strengths: efficient long-range processing with the retrieval power of attention. Watch this space — hybrid models are an active research area.

**Practical implication for agents:** SSMs may become the preferred architecture for agents with very long running contexts (multi-day tasks, persistent agents with weeks of history). For now, transformer-based models with aggressive context management remain the practical choice.

## 12.4 Context Window Limits and Practical Impact

| Context Window | Approximate Pages | Practical Limit |
|---|---|---|
| 4K tokens | ~3 pages | Short conversations, single documents |
| 32K tokens | ~25 pages | Multi-document research, long conversations |
| 128K tokens | ~100 pages | Book-length contexts, long agent sessions |
| 1M tokens | ~750 pages | Large codebases, entire document repositories |

**The practical ceiling is lower than the technical ceiling.** A 1M-token context model can technically accept 750 pages of input, but:
- Cost is linear (or super-linear) with input length
- Latency for first token increases with context
- Lost-in-the-middle effects mean middle content may not be attended to
- The quality gain from including all context may be marginal if only a small fraction is relevant

**For agents:** Context window size determines how much history, retrieved content, and tool output the agent can work with per call. Design context management assuming you will need to summarize and prune long before reaching the technical limit.

## 12.5 Open vs. Closed-Source Models for Agentic Deployment

| Dimension | Frontier-Closed (vendor API, frontier-tier) | Capable-Open (self-hostable open-weight) |
|---|---|---|
| Capability (frontier) | Higher (current frontier) | Lower for most tasks, closing gap |
| Deployment | API-only (vendor controls infra) | Self-hosted or managed (you control infra) |
| Data privacy | Data sent to vendor | Data stays on your infra |
| Latency (self-hosted) | Vendor-dependent | Controllable with your hardware |
| Cost at scale | High API costs | Compute costs (hardware or cloud GPU) |
| Customization (fine-tuning) | Limited, vendor-controlled | Full fine-tuning on your data |
| Availability/reliability | Vendor SLA | Your responsibility |
| Regulatory compliance | Vendor certifications | You certify your own infra |
| Tool use quality | Excellent (purpose-built) | Variable; improving |
Examples as of mid-2026. Frontier-Closed: Claude Opus, GPT Sol. Capable-Open: Llama, Mistral, Qwen's largest open-weight releases.

**When to use closed-source:** You need frontier capability, you're in the prototyping or low-volume phase, you don't have GPU infrastructure, or vendor SLAs are sufficient for your reliability requirements.

**When to use open-source:** Data privacy requirements prevent sending data to a vendor; you need to fine-tune on proprietary data; you are at high volume where API costs exceed self-hosted costs; regulatory requirements mandate on-premise processing.

**The hybrid approach:** Use closed-source models at the orchestrator level (where highest capability matters) and fine-tuned open-source models for high-volume, repeatable subagent tasks (where cost per task dominates). This is increasingly common in production at scale.

## 12.6 Test Yourself

**Q12.1** Explain the quadratic attention problem. At what context lengths does it become practically significant for production agent design?

**Q12.2** What is the "lost in the middle" problem and how does it affect agent design? What mitigation strategies would you use?

**Q12.3** Describe the tradeoff between transformer-based and SSM-based architectures for long-running agents. In what scenario would you actively consider a Mamba-based model?

**Q12.4** Your agent runs 20-step tasks with verbose tool outputs, consuming 80K tokens per run. The model has a 128K context window. Is this safe? What concerns would you have?

**Q12.5** Compare RoPE position embeddings to original absolute position encodings. Why does RoPE generalize better to longer sequences than the model was trained on?

**Q12.6** You are building an agent for a healthcare company that cannot send patient data to external vendors. What model deployment strategy would you recommend?

**Q12.7** At what point does switching from a closed-source API to a self-hosted open-source model become cost-effective? Describe the calculation.

**Q12.8** Explain prompt caching at the KV-cache level. Why does it reduce input token costs, and under what conditions does it not apply?

## Further reading

- Gu, A., Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". https://arxiv.org/abs/2312.00752
- Hong, K., Troynikov, A., Huber, J. (2025). "Context Rot: How Increasing Input Tokens Impacts LLM Performance". https://research.trychroma.com/context-rot
- Liu, N., Lin, K., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts". https://arxiv.org/abs/2307.03172
- Su, J., Lu, Y., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". https://arxiv.org/abs/2104.09864
- Vaswani, A., Shazeer, N., et al. (2017). "Attention Is All You Need". https://arxiv.org/abs/1706.03762
