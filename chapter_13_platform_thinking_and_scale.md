# Chapter 13: Platform Thinking and Scale

## 13.1 From Single Agent to Platform

An individual agent solving a single task is a tool. A platform is a shared system that multiple teams use to build and deploy agents, with common infrastructure for authentication, observability, tool access, policy enforcement, and model management.

Platform thinking applies when:
- Multiple teams want to build agents and you don't want each team reinventing the same infrastructure
- You need consistent policy enforcement (guardrails, safety, audit) across all agents
- You want to share expensive infrastructure (GPU capacity, vector databases, tool integrations) rather than duplicate it
- You need a single pane of glass for monitoring, cost attribution, and incident response

## 13.2 Multi-Tenant Architecture

A multi-tenant agent platform serves multiple teams (tenants) from the same infrastructure while maintaining isolation between them.

**Isolation requirements:**
- **Data isolation:** Tenant A's agent must not be able to read Tenant B's data, tool results, or conversation history.
- **Resource isolation:** Tenant A's high-volume workload should not degrade Tenant B's latency.
- **Policy isolation:** Tenant A's guardrail configuration should not affect Tenant B's agent behavior.
- **Billing isolation:** Cost must be attributed to the correct tenant.

**Implementation patterns:**
- **Namespace-scoped tool access:** Each tenant's tool catalog is a subset of the global tool catalog, filtered by tenant-granted permissions.
- **Tenant-scoped vector stores:** Each tenant has a separate namespace or index in the vector database; retrieval queries are scoped to the tenant's namespace by default.
- **Resource quotas:** Enforce per-tenant limits on tokens per day, concurrent agent sessions, and tool call rates.
- **Audit log partitioning:** Each tenant's audit log is stored in a separate partition that their administrators can access without seeing other tenants' logs.

## 13.3 Architectural Trade-offs at Platform Scale

| Decision          | Option A                          | Option B                         | When A Wins                  | When B Wins                              |
| ----------------- | --------------------------------- | -------------------------------- | ---------------------------- | ---------------------------------------- |
| **Model serving** | Centralized (one fleet)           | Per-tenant (dedicated)           | Cost efficiency, utilization | Strong isolation, compliance             |
| **Tool registry** | Global (all tenants share)        | Per-tenant (custom tools)        | Reduced duplication          | Tenant-specific integrations             |
| **Vector DB**     | Shared (namespaced)               | Per-tenant instance              | Cost at scale                | Strict isolation, per-tenant fine-tuning |
| **Guardrails**    | Centralized (platform enforces)   | Configurable (tenant customizes) | Consistent enforcement       | Tenant flexibility                       |
| **Observability** | Centralized (platform aggregates) | Per-tenant dashboards            | Cross-tenant fleet insight   | Tenant self-service monitoring           |

## 13.4 Research-to-Production Bridges

The gap between a research prototype and a production platform is often underestimated. Key transitions:

**From notebook to service:** The research prototype runs in a notebook. Production requires a service with an API, authentication, rate limiting, health checks, and deployment automation. This is standard software engineering, but it often surprises ML-focused teams.

**From single-run to fleet:** A prototype is evaluated on one task at a time. Production runs thousands of concurrent agent sessions. This surfaces concurrency bugs, resource contention, and latency distributions that were invisible at one-at-a-time scale.

**From happy-path to error handling:** Research prompts assume cooperative inputs and working tools. Production handles user inputs that are adversarial, ambiguous, or out-of-distribution; tools that are flaky, slow, or changed their API; and models that occasionally produce unexpected outputs.

**From offline eval to online monitoring:** Research uses a fixed eval set. Production requires a live monitoring system that detects behavioral drift before it causes significant harm.

## 13.5 Test Yourself

**Q13.1** What is a multi-tenant agent platform and what isolation requirements must it satisfy? Give a concrete example of an isolation failure and its consequence.

**Q13.2** A company has five teams that each want to build AI agents. Describe the shared infrastructure you would build to serve all five, and what you would leave for each team to own.

**Q13.3** How would you implement per-tenant tool access control in a shared agent platform? Describe the data model and the enforcement mechanism.

**Q13.4** Describe the three most important engineering transitions when taking an agent from a research prototype to a production service.

**Q13.5** A platform team is deciding between a shared vector database (namespaced per tenant) and per-tenant vector database instances. Walk through the tradeoffs and make a recommendation for a platform with 50 tenants and 5 million documents per tenant.

**Q13.6** An enterprise customer requires that their agents' data never comingle with other tenants' data, even at the storage layer. What architecture changes does this require, and what is the cost?

**Q13.7** How do you attribute token costs to individual tenants in a shared LLM infrastructure? What granularity would you track, and how would you surface this to tenants?

## Further reading

- AWS. "Well-Architected Framework: SaaS Lens". https://docs.aws.amazon.com/wellarchitected/latest/saas-lens/saas-lens.html
