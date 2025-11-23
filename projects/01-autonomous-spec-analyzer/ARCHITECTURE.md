# Architecture Decision Record (ADR)

## Project: Multi-Cloud Spec Analyzer

### Context

We needed a way to analyze requirements using the large context window of Google Gemini, but our reporting standard requires Azure OpenAI compliance.

### Decision

We implemented a **LangGraph State Machine** with a **Human-in-the-Loop** approval step.

### Architecture

1. **Ingestion Node:** Google Gemini 1.5 Pro (Vertex AI) analyzes the text for gaps.
2. **Transformation Node:** Azure GPT-4 formats the analysis into Jira JSON.
3. **Approval Node:** The graph pauses (Interrupt) for human verification.
4. **Persistence:** State is saved in memory to allow for loops/retries.

### Trade-offs

- **Latency:** Using two different clouds adds network latency, but allows us to leverage specific model strengths.
- **Cost:** Running GPT-4 is more expensive than Gemini Flash, but necessary for strict output formatting.
