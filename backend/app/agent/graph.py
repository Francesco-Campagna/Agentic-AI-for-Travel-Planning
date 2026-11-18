"""Grafo LangGraph dell'agente.

Un nodo router che classifica la richiesta.
Un agente che può chiamare dei tool. 
Memoria della conversazione in RAM tramite MemorySaver
(thread_id = session_id del frontend).

Grafo:
  START -> router ->  DIRETTA ->  smalltalk -> END
                   |   
                -> STRUMENTI -> agent -> tools -> agent -> ...   (si ripete finché serve)
                                   |                                         |
                                   |                                         v
                                   -> no tools ->         ->         (risposta finale) 
                                   
Teoricamente l'agente potrebbe non chiamare mai i tool, ma in pratica li chiama sempre almeno una volta (testato con modelli piccoli).
"""

from functools import lru_cache

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from ..config import settings
from .state import AgentState
from .tools import cerca_conoscenza, meteo_citta, cerca_voli

from .prompts import SMALLTALK_PROMPT, SYSTEM_PROMPT, get_system_prompt
from .router import needs_search

from typing import Literal
from langgraph.types import Command

# Tutti gli strumenti dell'agente. Meteo e voli verranno aggiunti qui.
TOOLS = [cerca_conoscenza, meteo_citta, cerca_voli]


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Client verso il modello Ollama locale.

    Il client viene creato una sola volta per processo.
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.ollama_temperature,
    )


@lru_cache(maxsize=1)
def get_agent_llm():
    """Lo stesso LLM, con i tool "collegati": è questa versione che decide se chiamarne uno.

    Usata sia dal nodo "agent" sia dal warm-up: i tool entrano nel prompt, quindi il
    warm-up deve usare la stessa versione per scaldare la cache del prompt giusto.
    """
    return get_llm().bind_tools(TOOLS)


def _compact_history(messages, keep_turns=3, tool_chars=300):
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_idx:
        return messages

    last_human = human_idx[-1]
    past_humans = human_idx[:-1]
    start = past_humans[-keep_turns] if len(past_humans) >= keep_turns else 0

    past = []
    for m in messages[start:last_human]:
        if isinstance(m, ToolMessage):
            text = str(m.content)
            if len(text) > tool_chars:
                text = text[:tool_chars] + " … [risultato abbreviato]"
            m = m.model_copy(update={"content": text})
        past.append(m)          # Human e AI (con o senza tool_calls) invariati

    return [*past, *messages[last_human:]]

def call_model(state: AgentState) -> dict:
    """Nodo "agent": chiama l'LLM con system prompt + cronologia.

    La risposta può essere testo finale oppure la richiesta di chiamare un tool.
    """
    
    history = _compact_history(state["messages"])
    has_results = isinstance(history[-1], ToolMessage)
    llm = get_llm() if has_results else get_agent_llm()
    return {"messages": [llm.invoke([SystemMessage(content=get_system_prompt()), *history])]}

def router_node(state: AgentState) -> Command[Literal["agent", "smalltalk"]]:
    route = "agent" if needs_search(get_llm(), state["messages"]) else "smalltalk"
    return Command(update={"route": route}, goto=route)


def smalltalk(state: AgentState) -> dict:
    """Risponde con il modello SENZA tool, quindi non può chiamare la ricerca.

    Passa solo il testo scambiato (messaggi utente e risposte finali), senza le
    chiamate ai tool dei turni precedenti, che un modello piccolo tenderebbe a ripetere.
    """
    history = [
        m
        for m in state["messages"]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content and not m.tool_calls)
    ]
    response = get_llm().invoke([SystemMessage(content=SMALLTALK_PROMPT), *history])
    return {"messages": [response]}


_checkpointer = MemorySaver()


@lru_cache(maxsize=1)
def build_graph():
    """Costruisce e compila il grafo.

    La memoria è in RAM (MemorySaver): si azzera a ogni riavvio del backend.
    Per una persistenza reale si può sostituire con un checkpointer su file
    o Db, senza cambiare la struttura del grafo.
    """
    graph = StateGraph(AgentState)
    
    graph.add_node("router", router_node)
    graph.add_node("smalltalk", smalltalk)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))  # esegue i tool richiesti dal modello

    # START -> router
    graph.add_edge(START, "router")
    
    graph.add_edge("smalltalk", END)
    # Se l'ultimo messaggio contiene richieste di tool va a "tools", altrimenti a END.
    graph.add_conditional_edges("agent", tools_condition)  # dopo il modello, se ha chiesto un tool va a tools, altrimenti a END
    graph.add_edge("tools", "agent")  # dopo il tool, il modello legge il risultato e risponde
    return graph.compile(checkpointer=_checkpointer)