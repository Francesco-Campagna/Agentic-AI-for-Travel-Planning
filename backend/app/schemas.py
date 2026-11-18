"""Schemi delle richieste e risposte dell'API.

Il contratto con il frontend è:
  POST /api/chat  { session_id, message }
  -> { answer: str, sources?: [{ title, source }], tools?: [str] }
  GET /api/history/{session_id}
  -> { messages: [{ role: "user" | "agent", text: str }] }

"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Id di sessione lato browser, usato come thread_id di LangGraph")
    message: str = Field(..., description="Messaggio dell'utente")


class SourceItem(BaseModel):
    title: str
    source: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    tools: list[str] = []


class HistoryMessage(BaseModel):
    role: Literal["user", "agent"]
    text: str
    sources: list[SourceItem] = []
    tools: list[str] = []


class HistoryResponse(BaseModel):
    """Cronologia di una sessione, usata dal frontend per ripristinare la chat."""

    messages: list[HistoryMessage] = []


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_base_url: str
    ollama_reachable: bool
