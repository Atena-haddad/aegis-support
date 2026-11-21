from pydantic import BaseModel, Field


class AgentDraftResponse(BaseModel):
    """Structured output contract for the Resolution Agent."""
    resolution_steps: str = Field(
        description="Clear, actionable engineering/support resolution grounded ONLY in the retrieved SOP chunks."
    )
    cited_doc_ids: list[str] = Field(
        description="Exact doc_id strings (e.g. 'api_errors_runbook.md#L2') used to support the answer."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 that the SOP context completely answers the ticket.",
    )
    needs_human_escalation: bool = Field(
        description="True if the retrieved SOP documents do NOT contain the answer or if manual Ops intervention is required."
    )


class CriticAuditResult(BaseModel):
    """Structured output contract for the Hallucination & Grounding Critic Agent."""
    is_grounded: bool = Field(
        description="True ONLY if every technical claim and step in the draft is directly supported by the retrieved chunks."
    )
    critique: str = Field(
        description="Brief explanation of any unsupported claim or confirmation that all claims are verified."
    )
    refined_search_query: str = Field(
        description="If is_grounded is False, provide a sharper keyword/semantic query to retry retrieval; otherwise empty string."
    )