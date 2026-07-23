# Chapter 5: RAG (Retrieval-Augmented Generation) for Agents

## 5.1 RAG, Agentic RAG, and Self-RAG

**Retrieval-Augmented Generation (RAG)** addresses the fundamental limitation that models cannot know what they weren't trained on: recent events, proprietary documents, or domain-specific data. In RAG, before the model generates a response, the system retrieves relevant documents from an external corpus and includes them in the model's context.

The classical RAG pipeline:
```
Query → Embed query → Search vector store → Return top-K chunks → 
Prepend to prompt → Generate response
```

This is a single-turn, fixed retrieval pipeline. It works well for knowledge-grounded Q&A but is rigid: one retrieval step, fixed K, no feedback.

**Agentic RAG** treats retrieval as a tool. The agent decides when to retrieve, what to retrieve, and how many times to retrieve — rather than having retrieval hard-coded before every generation. This allows:
- Iterative retrieval: retrieve, read, identify gaps, retrieve again with a refined query
- Conditional retrieval: only retrieve when the model determines it lacks the necessary information
- Multi-source retrieval: query different document stores depending on what kind of information is needed

**Self-RAG** (Asai et al., 2023) goes further: the model itself decides at generation time whether retrieval is needed for the current segment of its output, retrieves if yes, evaluates the retrieved passages for relevance and supportiveness, and generates with or without those passages. Self-RAG adds special tokens that the model generates to signal retrieval decisions and critique passages inline with output generation.

| System | Who Decides to Retrieve | How Often | Feedback on Retrieval Quality |
|---|---|---|---|
| **Classical RAG** | Always (hard-coded) | Once per query | No |
| **Agentic RAG** | The agent, as needed | Multiple times | Indirect (agent observes and decides) |
| **Self-RAG** | The model itself (via special tokens) | Per segment | Yes (relevance/support tokens inline) |

In practice, Agentic RAG is the most commonly used pattern in production because it is more flexible than classical RAG and simpler to deploy than Self-RAG (which requires a specially trained model or post-training).

## 5.2 Chunking Strategies

Before anything can be retrieved, documents must be chunked: divided into pieces small enough to fit in a context window alongside other content, while remaining semantically coherent.

**Fixed-size chunking:** Split every N tokens with optional overlap (e.g., 512 tokens, 50-token overlap). Simple, fast, consistent. The main weakness is that it splits at semantically arbitrary points, sometimes breaking sentences or concepts across chunks.

**Sentence-level chunking:** Split at sentence boundaries. Preserves semantic units. Works well for short, well-structured documents. Can produce highly variable chunk sizes.

**Recursive character-level text splitting:** Try to split on paragraph boundaries first; if a paragraph is too long, split on sentence boundaries; if still too long, split on word boundaries. This is the approach used by LangChain's `RecursiveCharacterTextSplitter`. More coherent than fixed-size for prose.

**Semantic chunking:** Embed sentences and cluster them; split where semantic similarity between adjacent sentences drops below a threshold. Produces the most semantically coherent chunks but is computationally expensive and harder to debug.

**Document-structure-aware chunking:** Split on document-native structure: headings, sections, tables, code blocks. Best for structured documents (PDFs with headers, HTML, Markdown). Requires document structure to be parseable.

**Hierarchical chunking (parent-child):** Store small chunks (sentences or short paragraphs) for precision retrieval, but also store their parent section or document. When a small chunk is retrieved, inject the parent chunk into context for broader context. This is one of the most effective patterns for complex documents.

| Strategy             | Coherence   | Size Consistency | Compute Cost | Best For                                          |
| -------------------- | ----------- | ---------------- | ------------ | ------------------------------------------------- |
| **Fixed-size**           | Low         | High             | Low          | Large-scale indexing where precision is secondary |
| **Sentence-level**       | Medium      | Low              | Low          | Short, well-structured documents                  |
| **Recursive text split** | Medium-high | Medium           | Low          | Prose documents                                   |
| **Semantic**             | High        | Low              | High         | High-precision applications                       |
| **Structure-aware**      | High        | Medium           | Medium       | Structured docs (PDFs, HTML)                      |
| **Hierarchical**         | High        | Medium           | Medium       | Complex multi-section documents                   |

## 5.3 Hybrid Search: Dense + Sparse (BM25)

**Dense retrieval** uses embedding models to encode queries and documents as vectors, then retrieves by cosine similarity. It captures semantic meaning: "car" and "automobile" return similar results. It works well for paraphrase, synonym handling, and concept-level queries.

**Sparse retrieval (BM25)** is a keyword-based ranking function that scores documents by term frequency (how often the query term appears) and inverse document frequency (how rare the term is across all documents). BM25 excels at exact-match retrieval: if the user queries for a specific product SKU, a specific error code, or a specific person's name, BM25 will find it reliably even if the dense embedding fails to distinguish it from similar-looking strings.

For a query term `q` in document `d`, BM25's score contribution is:

```
score(q, d) = IDF(q) × [ f(q, d) × (k1 + 1) ] / [ f(q, d) + k1 × (1 - b + b × |d| / avgdl) ]
```

where `f(q, d)` is the term's frequency in `d`, `|d|` is the document length, `avgdl` is the average document length in the corpus, and `k1` and `b` are tuning constants controlling term-frequency saturation and length normalization, respectively. In other words, BM25 is TF-IDF with two refinements: it caps the benefit of repeating a term too many times (saturation), and it penalizes long documents that rack up matches simply by being long (length normalization).

**Hybrid search** combines both: retrieve top-K from dense search and top-K from BM25, then merge and re-rank the combined set. The most common merging strategy is Reciprocal Rank Fusion (RRF), which combines rank positions across the two lists without requiring score normalization.

**When dense wins:** Conceptual, paraphrase, or natural-language queries ("how do I reset my password?" matches "account recovery instructions").

**When BM25 wins:** Exact-match queries (error codes, model numbers, names, IDs), technical jargon not in the embedding model's training data, very short queries.

**In practice,** hybrid search consistently outperforms either approach alone across most retrieval benchmarks. The additional latency is small (BM25 is fast), and the quality gain is significant, especially for heterogeneous document sets.

## 5.4 Re-ranking

After retrieval, the top-K chunks are ranked by similarity score — which reflects embedding proximity, not relevance to the actual task. A re-ranker applies a more expensive model to re-score the top-K results and re-order them.

**Cross-encoders** are the standard re-ranker architecture. Unlike bi-encoders (which encode query and document separately), cross-encoders process the query and document together as a single input, which allows the model to directly compute relevance. This is slower (O(K) forward passes) but more accurate.

**Where re-ranking adds value:** Dense retrieval can return semantically similar but task-irrelevant documents (e.g., a query about "model performance" might retrieve documents about car performance). A re-ranker can demote these. Re-ranking is most valuable when the retrieval set is large (top-50) and precision in the final top-5 matters.

**Typical pipeline:**

```
Query → Embed → Dense search (top-50) + BM25 (top-50) → RRF merge (top-50) → 
Re-rank (cross-encoder, top-10) → Final context (top-5)
```

**Cost consideration:** Re-ranking scores each retrieved chunk against the query. For 50 chunks, this is 50 query-document scoring passes, typically batched into a few GPU calls rather than issued as 50 separate requests. Cross-encoders are typically smaller models (hundreds of millions of parameters), so this is manageable, but it adds 50-200ms to the pipeline.

## 5.5 Query Transformation

The query the user sends is often not the best query for retrieval. Query transformation techniques rewrite or expand the query before retrieval to improve recall and precision.

**HyDE (Hypothetical Document Embeddings):** Instead of embedding the query directly, generate a hypothetical document that would answer the query (using the model), then embed that document. The intuition: a real document answering the question lives in a different semantic space than the question itself, and the hypothetical document is a better proxy. Works well when query-document style mismatch is significant.

**Query decomposition:** Break a complex query into sub-queries, retrieve for each, and merge results. Example: "What are the pros and cons of PostgreSQL vs. MySQL for write-heavy workloads?" becomes two separate queries plus a synthesis step.

**Step-back prompting:** Generate a more abstract version of the query before retrieval. "Which cloud provider should I use for my startup?" becomes "What factors matter for cloud provider selection?" — which retrieves foundational content rather than opinion pieces.

**Query expansion:** Augment the original query with synonyms, related terms, or alternative phrasings generated by the model. Improves recall but risks diluting precision.

**Multi-query generation:** Generate N rephrasings of the original query, retrieve for each, deduplicate, and merge. Simple ensemble that improves recall significantly with low overhead.

| Technique | When to Use | Main Risk |
|---|---|---|
| **HyDE** | Query-document style mismatch (short questions, long docs) | Hypothetical doc can introduce hallucinated content into the embedding |
| **Query decomposition** | Multi-part or multi-hop questions | Sub-query answers may not synthesize cleanly |
| **Step-back prompting** | Abstract knowledge questions | May retrieve too-general content |
| **Query expansion** | Narrow vocabulary, synonym-heavy domains | Precision loss from added noise terms |
| **Multi-query** | General recall improvement | Multiple retrieval calls multiply latency and cost |

## 5.6 Hallucination Mitigation in RAG

RAG reduces but does not eliminate hallucination. Even when grounded documents are present, models can misread them, over-generalize from them, or generate confident text that subtly contradicts the retrieved content.

**Source attribution:** Require the model to cite the specific chunk or passage that supports each claim. This does not prevent hallucination but makes it detectable and auditable.

**Faithfulness evaluation:** After generation, use a separate model or prompt to check whether each claim in the response is supported by the retrieved chunks. Return "not found in provided context" rather than a synthesized answer when support is absent.

**Retrieval-then-answer decomposition:** Separate the retrieval and generation steps with an explicit "can I answer this from the retrieved context?" check before generating the response.

**Reducing context noise:** Including irrelevant documents in context increases hallucination. Better retrieval precision (see hybrid search and re-ranking) reduces the number of irrelevant chunks the model must work around.

**Temperature reduction:** Lower generation temperature reduces variation and tends to produce more faithful responses, at the cost of less paraphrase. For RAG-grounded answers, temperature 0 or near-0 is often appropriate.

**Chunk quality control:** Hallucination often traces to poorly chunked content — truncated sentences, missing context, tables split across chunks. Improving chunking quality is one of the highest-leverage hallucination mitigations.

## 5.7 Scaling RAG to Millions of Documents

A RAG system handling thousands of queries per day over millions of documents has engineering challenges beyond what works in a proof of concept.

**Indexing infrastructure:** Embedding millions of documents requires a distributed embedding pipeline. A common pattern: batch-process documents with a smaller, faster embedding model (e.g., a 100M-parameter model); store embeddings in a dedicated vector database (Pinecone, Weaviate, Qdrant, pgvector).

**ANN vs. exact search:** Exact nearest-neighbor search over millions of vectors is too slow for real-time retrieval. Approximate nearest neighbor (ANN) indices (HNSW, IVF-PQ) trade a small recall penalty for large speedups (milliseconds instead of seconds). HNSW is the most common choice for its balance of recall and latency.

**Sharding and partitioning:** Distribute the document index across shards by topic, document type, date, or user segment. Queries are routed to the appropriate shard(s), reducing the search space per query.

**Caching:** Cache embeddings for common queries. Cache retrieval results for repeated queries. For a document corpus that changes slowly, retrieval results may be valid for hours or days.

**Metadata filtering:** Use metadata filters to pre-scope retrieval before vector search. If the user is asking about "financial reports from 2023," filter to documents with `year=2023` and `type=financial` before running the vector search. This dramatically reduces the search space and improves precision.

**Incremental indexing:** New documents should be indexed without re-embedding the entire corpus. Vector databases support incremental insertion; the main engineering challenge is handling document updates (the old embedding must be deleted or flagged stale).

**Cost management:** Embedding API calls and vector DB queries have real costs at scale. Strategies: batch embedding during off-peak hours, use smaller embedding models for initial indexing and re-embed with larger models only for the top candidates, cache aggressively.

## 5.8 Test Yourself

**Q5.1** Explain the difference between classical RAG, Agentic RAG, and Self-RAG to a senior engineer who knows ML but hasn't worked on agents. When would you choose each?

**Q5.2** A team uses fixed-size chunking (512 tokens) for a legal document corpus and is seeing poor retrieval quality. Diagnose what is likely going wrong and describe what chunking strategy you would switch to.

**Q5.3** You are building a RAG system for a technical support chatbot. Users ask both vague questions ("it's slow") and precise ones ("Error code E502 on product model XR-7"). Why does pure dense retrieval fail for both, and how does hybrid search address it?

**Q5.4** When would you use HyDE, and what is its main risk? Describe a retrieval scenario where HyDE would clearly outperform standard query embedding.

**Q5.5** A re-ranker improves your RAG pipeline's precision from 0.62 to 0.81 in offline evaluation. Is it worth deploying? What factors would you weigh?

**Q5.6** Your RAG system has high retrieval recall but the model still produces hallucinated answers. Walk through the diagnostic process you would follow to identify the cause.

**Q5.7** Describe how you would scale a RAG system from 10,000 to 10 million documents. Which components change and how?

**Q5.8** What is Reciprocal Rank Fusion? When is it used and why is it preferable to score-based fusion?

**Q5.9** You need to retrieve information from a 500-page technical manual to answer a user's precise question. Describe your full RAG pipeline: chunking, indexing, retrieval, re-ranking, and generation. Justify each choice.

**Q5.10** A user asks: "What did the CEO say about revenue guidance last quarter?" over a corpus of 5,000 earnings call transcripts. Identify three retrieval challenges this query poses and how you would address each.

**Q5.11** Explain the tradeoff between retrieval context length and answer quality. When does adding more retrieved chunks hurt rather than help?

**Q5.12** Describe query decomposition and give a worked example. What are the failure modes you need to handle in the synthesis step?

**Q5.13** (Identify the failure point.) A support agent answered "Your XR-7 is covered under the 2-year warranty" when the correct answer, per policy, is a 1-year warranty. The retrieval trace is below. Pinpoint where the pipeline failed, name the failure class, and give the fix.

```
User query: "What is the warranty period for the XR-7?"
Embedded query → dense search (top_k=4), no metadata filter, no re-ranker:
  [0.71] doc#1183  "...the XR-9 flagship ships with our extended 2-year warranty..."
  [0.69] doc#0044  "...warranty terms vary by product line; see the product page..."
  [0.68] doc#2210  "...the XR-7 warranty period is one (1) year from purchase..."
  [0.67] doc#0091  "...2-year warranty available as a paid add-on for all models..."
Generation prompt used top 2 chunks only (context-budget cap).
Model output: "Your XR-7 is covered under the 2-year warranty."
```

## Further reading

- Singh, A., Ehtesham, A., et al. (2025). "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" https://arxiv.org/abs/2501.09136
