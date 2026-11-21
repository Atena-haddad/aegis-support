import os
import time
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from src.ml_router.predict import MLTriageRouter
from src.rag.hybrid_retriever import HybridRetriever
from src.guardrails.pii import PIIGuardrail
from src.agent.schemas import AgentDraftResponse, CriticAuditResult

load_dotenv()


class TicketState(TypedDict):
    raw_query: str
    sanitized_query: str
    redacted_pii: list[str]
    # Classical ML Router outputs
    intent: str
    intent_confidence: float
    priority: str
    is_ood_novel: bool
    route_to_agent: bool
    routing_reason: str
    ml_latency_ms: float
    # Agentic RAG & Critic loop state
    search_query: str
    retrieved_chunks: list[dict]
    draft_answer: str
    cited_doc_ids: list[str]
    confidence: float
    is_grounded: bool
    critique: str
    retry_count: int
    escalate_to_human: bool
    execution_path: str
    total_latency_ms: float


class AegisAgentOrchestrator:
    def __init__(self):
        self.guardrail = PIIGuardrail()
        shared_embedder = __import__("fastembed").TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.router = MLTriageRouter(embedder=shared_embedder)
        self.retriever = HybridRetriever(embedder=shared_embedder)

        primary_model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite")

        primary_llm = ChatGoogleGenerativeAI(
            model=primary_model,
            temperature=0.0,
            max_retries=0,
        )
        fallback_llm = ChatGoogleGenerativeAI(
            model=fallback_model,
            temperature=0.0,
            max_retries=0,
        )

        # Production Resilience Pattern: Automatic model fallback on 503/429 spikes
        self.drafter_llm = primary_llm.with_structured_output(
            AgentDraftResponse, method="json_schema"
        ).with_fallbacks([
            fallback_llm.with_structured_output(AgentDraftResponse, method="json_schema")
        ])

        self.critic_llm = primary_llm.with_structured_output(
            CriticAuditResult, method="json_schema"
        ).with_fallbacks([
            fallback_llm.with_structured_output(CriticAuditResult, method="json_schema")
        ])
        self.graph = self._build_graph()

    # --- Graph Nodes ---
    def triage_node(self, state: TicketState) -> dict:
        sanitized = self.guardrail.sanitize(state["raw_query"])
        decision = self.router.route(sanitized.clean_text)
        return {
            "sanitized_query": sanitized.clean_text,
            "redacted_pii": sanitized.redacted_entities,
            "intent": decision.intent,
            "intent_confidence": decision.intent_confidence,
            "priority": decision.priority,
            "is_ood_novel": decision.is_ood_novel,
            "route_to_agent": decision.route_to_agent,
            "routing_reason": decision.routing_reason,
            "ml_latency_ms": decision.latency_ms,
            "search_query": sanitized.clean_text,
            "retry_count": 0,
        }

    def fast_path_node(self, state: TicketState) -> dict:
        # Retrieve top SOP template without calling an LLM (zero token cost)
        hits = self.retriever.retrieve(state["sanitized_query"], top_k=1)
        if hits:
            top = hits[0]
            answer = f"Automated Fast-Path Resolution (No LLM invoked): {top.content}"
            citations = [top.doc_id]
        else:
            answer = "Your routine request has been logged and queued for standard processing."
            citations = []

        return {
            "retrieved_chunks": [h.__dict__ for h in hits],
            "draft_answer": answer,
            "cited_doc_ids": citations,
            "confidence": 0.95,
            "is_grounded": True,
            "critique": "Bypassed LLM via Classical ML Fast-Path.",
            "escalate_to_human": False,
            "execution_path": "CLASSICAL_ML_FAST_PATH",
        }

    def retrieve_and_draft_node(self, state: TicketState) -> dict:
        hits = self.retriever.retrieve(state["search_query"], top_k=3)
        context_block = "\n".join(
            [f"- [{h.doc_id}] (rerank={h.rerank_score}): {h.content}" for h in hits]
        )

        prompt = (
            "You are AegisSupport, an enterprise AI reliability & operations engineer.\n"
            "Answer the ticket using ONLY the retrieved SOP context below. "
            "Do NOT invent commands, SLAs, or policies not present in the context. "
            "If the context does not cover the issue, set needs_human_escalation=True.\n\n"
            f"Ticket Intent: {state['intent']} | Priority: {state['priority']} | Novel/OOD: {state['is_ood_novel']}\n"
            f"User Ticket: {state['sanitized_query']}\n\n"
            f"Retrieved SOP Chunks:\n{context_block}"
        )
        draft: AgentDraftResponse = self.drafter_llm.invoke(prompt)
        return {
            "retrieved_chunks": [h.__dict__ for h in hits],
            "draft_answer": draft.resolution_steps,
            "cited_doc_ids": draft.cited_doc_ids,
            "confidence": draft.confidence,
            "escalate_to_human": draft.needs_human_escalation,
            "execution_path": "AGENTIC_RAG_WITH_CRITIC",
        }

    def critic_verify_node(self, state: TicketState) -> dict:
        context_block = "\n".join(
            [f"- [{c['doc_id']}]: {c['content']}" for c in state["retrieved_chunks"]]
        )
        prompt = (
            "You are a strict AI safety and grounding auditor.\n"
            "Verify whether EVERY claim in the Draft Answer is directly supported by the Retrieved SOP Chunks.\n"
            "If the draft invents facts not in the chunks, or if the ticket is completely unrelated to the chunks, "
            "set is_grounded=False and provide a refined_search_query.\n\n"
            f"User Ticket: {state['sanitized_query']}\n"
            f"Draft Answer: {state['draft_answer']}\n"
            f"Retrieved SOP Chunks:\n{context_block}"
        )
        audit: CriticAuditResult = self.critic_llm.invoke(prompt)
        new_retry = state["retry_count"] + 1
        should_escalate = state["escalate_to_human"] or (not audit.is_grounded and new_retry > 1)

        return {
            "is_grounded": audit.is_grounded,
            "critique": audit.critique,
            "search_query": audit.refined_search_query or state["search_query"],
            "retry_count": new_retry,
            "escalate_to_human": should_escalate,
        }

    # --- Conditional Edges ---
    def _route_after_triage(self, state: TicketState) -> Literal["fast_path", "retrieve_and_draft"]:
        return "retrieve_and_draft" if state["route_to_agent"] else "fast_path"

    def _route_after_critic(self, state: TicketState) -> Literal["retrieve_and_draft", "__end__"]:
        if not state["is_grounded"] and state["retry_count"] <= 1 and not state["is_ood_novel"]:
            return "retrieve_and_draft"
        return "__end__"

    def _build_graph(self):
        workflow = StateGraph(TicketState)
        workflow.add_node("triage", self.triage_node)
        workflow.add_node("fast_path", self.fast_path_node)
        workflow.add_node("retrieve_and_draft", self.retrieve_and_draft_node)
        workflow.add_node("critic_verify", self.critic_verify_node)

        workflow.add_edge(START, "triage")
        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {"fast_path": "fast_path", "retrieve_and_draft": "retrieve_and_draft"},
        )
        workflow.add_edge("fast_path", END)
        workflow.add_edge("retrieve_and_draft", "critic_verify")
        workflow.add_conditional_edges(
            "critic_verify",
            self._route_after_critic,
            {"retrieve_and_draft": "retrieve_and_draft", "__end__": END},
        )
        return workflow.compile()

    def run(self, raw_query: str) -> dict:
        start = time.perf_counter()
        result = self.graph.invoke({"raw_query": raw_query})
        result["total_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result


if __name__ == "__main__":
    orchestrator = AegisAgentOrchestrator()

    tests = [
        # 1. Fast-Path (Classical ML handles without LLM)
        "How do I download VAT invoices for accounting?",
        # 2. Complex Bug + PII Redaction + Critic Verification
        "Urgent: user atena@company.com with key ags_live_98765 gets HTTP 502 Bad Gateway on inference endpoint.",
        # 3. Out-of-Distribution / Unknown Issue -> Safe Escalation
        "Our quantum satellite telemetry uplink is desynchronizing with the orbital relay.",
    ]

    for q in tests:
        print("\n" + "=" * 70)
        out = orchestrator.run(q)
        print(f"INPUT:      {q}")
        print(f"SANITIZED:  {out['sanitized_query']} (Redacted: {out['redacted_pii']})")
        print(f"PATH:       {out['execution_path']} | Reason: {out['routing_reason']}")
        print(f"GROUNDED:   {out['is_grounded']} | Escalate: {out['escalate_to_human']} | Citations: {out['cited_doc_ids']}")
        print(f"ANSWER:     {out['draft_answer']}")
        print(f"CRITIQUE:   {out['critique']} | Total Latency: {out['total_latency_ms']} ms")