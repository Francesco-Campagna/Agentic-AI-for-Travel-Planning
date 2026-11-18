"""API del Travel Planning Agent.

Endpoint:
  POST /api/chat    -> conversa con l'agente (contratto atteso dal frontend)
  GET  /api/history/{session_id} -> cronologia della sessione (per ripristinare la chat)
  GET  /api/health  -> stato del backend e raggiungibilità di Ollama
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from .flights.db import ensure_ready as init_flights_db

from .agent.graph import build_graph, get_agent_llm
from .agent.prompts import SYSTEM_PROMPT
from .config import settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    SourceItem,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel-agent")


async def _warm_up_model() -> None:
    """Fa leggere al modello system prompt e definizione dei tool prima della prima domanda."""
    try:
        await asyncio.to_thread(
            get_agent_llm().invoke,  # con i tool collegati, come nelle richieste vere
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="Ciao")],
            options={"num_predict": 1, "temperature": settings.ollama_temperature},
        )
        logger.info("Warm-up completato.")
    except Exception:
        logger.warning("Warm-up non riuscito, il modello si caricherà alla prima richiesta.", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_flights_db()
    asyncio.create_task(_warm_up_model())
    yield


app = FastAPI(title="Travel Planning Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Il grafo viene compilato una volta sola all'avvio del processo.
graph = build_graph()


def _ollama_unreachable_message(detail: str) -> str:
    return (
        f"Impossibile contattare Ollama su {settings.ollama_base_url} con il modello "
        f"'{settings.ollama_model}'. Verifica che Ollama sia avviato e che il modello sia "
        f"stato scaricato (comando: ollama pull {settings.ollama_model}). Dettaglio: {detail}"
    )


def _tool_info(messages: list) -> tuple[list[SourceItem], list[str]]:
    """Estrae fonti e nomi dei tool dai messaggi "tool" di una lista di messaggi.

    Ogni risultato di un tool porta con sé: .name (quale tool) e .artifact (le fonti
    restituite da cerca_conoscenza, che non sono mai state mostrate al modello).
    """
    sources: list[SourceItem] = []
    tools: list[str] = []
    seen: set[tuple[str, str]] = set()
    for m in messages:
        if m.type != "tool":
            continue
        if m.name and m.name not in tools:
            tools.append(m.name)
        for s in m.artifact or []:
            key = (s["title"], s["source"])
            if key not in seen:
                seen.add(key)
                sources.append(SourceItem(**s))
    return sources, tools


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_reachable = False
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        ollama_reachable = resp.status_code == 200
    except httpx.HTTPError:
        ollama_reachable = False

    return HealthResponse(
        status="ok",
        model=settings.ollama_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_reachable=ollama_reachable,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Il messaggio è vuoto.")

    # recursion_limit: massimo di passaggi nel grafo (agent e tools contano uno ciascuno).
    # Evita che un modello piccolo richiami lo stesso tool all'infinito.
    config = {"configurable": {"thread_id": req.session_id}, "recursion_limit": 8}

    try:
        result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    except GraphRecursionError as exc:
        logger.exception("L'agente ha superato il limite di passaggi")
        raise HTTPException(
            status_code=500,
            detail="L'agente ha fatto troppi tentativi senza arrivare a una risposta. Prova a riformulare la domanda.",
        ) from exc
    except Exception as exc:  # qualunque errore di connessione/modello verso Ollama
        logger.exception("Errore durante l'invocazione del grafo")
        raise HTTPException(status_code=503, detail=_ollama_unreachable_message(str(exc))) from exc

    messages = result["messages"]  # tutta la conversazione, non solo questo turno
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    sources, tools = _tool_info(messages[last_human + 1 :])  # solo i tool usati in questo turno

    last_message = messages[-1]
    answer = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
    if not answer.strip():  # i modelli piccoli a volte chiudono con un messaggio vuoto
        answer = "Non sono riuscito a formulare una risposta. Puoi riprovare a riformulare la domanda?"

    return ChatResponse(answer=answer, sources=sources, tools=tools)


@app.get("/api/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str) -> HistoryResponse:
    """Restituisce i messaggi già scambiati in una sessione.

    Li legge dallo stato del grafo (checkpointer di LangGraph), la stessa
    memoria che l'agente usa per rispondere.
    Una sessione sconosciuta restituisce semplicemente una lista vuota.
    """
    state = graph.get_state({"configurable": {"thread_id": session_id}})
    stored = (state.values or {}).get("messages", [])

    items: list[HistoryMessage] = []
    turn_tool_messages: list = []  # risultati dei tool del turno in corso
    for m in stored:
        if m.type == "human":
            turn_tool_messages = []
            text = m.content if isinstance(m.content, str) else str(m.content)
            if text.strip():
                items.append(HistoryMessage(role="user", text=text))
        elif m.type == "tool":
            turn_tool_messages.append(m)  # non si mostra, ma ne ricaviamo fonti e strumenti
        elif m.type == "ai":
            if getattr(m, "tool_calls", None):  # passaggio intermedio: "chiamo il tool"
                continue
            text = m.content if isinstance(m.content, str) else str(m.content)
            if text.strip():
                sources, tools = _tool_info(turn_tool_messages)
                items.append(HistoryMessage(role="agent", text=text, sources=sources, tools=tools))
    return HistoryResponse(messages=items)