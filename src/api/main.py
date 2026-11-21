from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agent.graph import AegisAgentOrchestrator

orchestrator: AegisAgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    orchestrator = AegisAgentOrchestrator()
    yield


app = FastAPI(
    title="AegisSupport — Hybrid ML & Agentic Triage API",
    description=(
        "Enterprise ticket triage combining Classical ML/NLP routing, OOD novelty detection, "
        "Hybrid RAG (BM25 + Dense + Cross-Encoder), PII Guardrails, and a Self-Correcting LangGraph Agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class TicketRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["HTTP 502 Bad Gateway on inference endpoint"])


class TicketResponse(BaseModel):
    sanitized_query: str
    redacted_pii: list[str]
    intent: str
    intent_confidence: float
    priority: str
    is_ood_novel: bool
    execution_path: str
    routing_reason: str
    answer: str
    cited_doc_ids: list[str]
    confidence: float
    is_grounded: bool
    critique: str
    escalate_to_human: bool
    ml_latency_ms: float
    total_latency_ms: float


@app.get("/")
def root():
    return {
        "service": "AegisSupport — Hybrid ML & Agentic Triage API",
        "status": "live",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "aegis-support"}


@app.post("/triage", response_model=TicketResponse)
def triage_ticket(payload: TicketRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    result = orchestrator.run(payload.query)
    return TicketResponse(
        sanitized_query=result["sanitized_query"],
        redacted_pii=result["redacted_pii"],
        intent=result["intent"],
        intent_confidence=result["intent_confidence"],
        priority=result["priority"],
        is_ood_novel=result["is_ood_novel"],
        execution_path=result["execution_path"],
        routing_reason=result["routing_reason"],
        answer=result["draft_answer"],
        cited_doc_ids=result["cited_doc_ids"],
        confidence=result["confidence"],
        is_grounded=result["is_grounded"],
        critique=result["critique"],
        escalate_to_human=result["escalate_to_human"],
        ml_latency_ms=result["ml_latency_ms"],
        total_latency_ms=result["total_latency_ms"],
    )