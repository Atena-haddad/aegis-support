# AegisSupport — Hybrid Classical-ML & Agentic LLM Triage Engine

An enterprise technical support and incident triage system combining **Classical ML/NLP routing** and **Out-of-Distribution (OOD) novelty detection** with a **self-correcting LangGraph multi-agent RAG pipeline**, **PII privacy guardrails**, and **automated evaluation benchmarks**.

## Architecture

```text
Incoming Ticket (Text + Metadata)
       │
       ▼
[1. Pre-LLM Privacy Guardrail] ──► Regex/NER redaction (Emails, API keys, Stripe IDs, IPv4)
       │
       ▼
[2. Classical ML & NLP Router] ──► Sentence-Transformers (all-MiniLM-L6-v2) + Logistic Regression + IsolationForest
   • Intent & Priority classification + OOD novelty scoring
       │
       ├──► (Routine / Known Request) ──► Fast-Path SOP Template (0 LLM tokens)
       │
       └──► (Complex / High-Priority / OOD) ──► [3. Stateful LangGraph Agent]
                                                     │
                                                     ├──► Hybrid Retriever (BM25 + Dense + Cross-Encoder Reranker)
                                                     ├──► Drafter Node (Pydantic JSON Schema Enforcement)
                                                     └──► Critic Node (Grounding Audit + Conditional Retry / Escalation)

