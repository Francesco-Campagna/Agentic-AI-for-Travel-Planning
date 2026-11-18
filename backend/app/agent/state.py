"""Stato condiviso tra i nodi del grafo LangGraph.

"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
    route: str
