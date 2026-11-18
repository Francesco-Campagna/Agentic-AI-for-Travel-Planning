"""Decisione del turno: per rispondere serve usare uno strumento (knowledge base, meteo o voli)?

La decide il modello stesso, con una domanda stretta e senza tool collegati.
Fail-safe: se la risposta non è chiaramente "DIRETTA" (o il modello va in errore)
si assume che servano gli strumenti.
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger("travel-agent")

ROUTER_PROMPT = """Sei un classificatore per l'assistente di un'agenzia di viaggio.
Devi decidere se l'ULTIMO messaggio dell'utente richiede di consultare uno strumento
(knowledge base su destinazioni, attrazioni, orari, trasporti, itinerari, policy, rimborsi,
voli, bagagli, diritti dei passeggeri; oppure meteo di una città).

Rispondi con UNA sola parola: STRUMENTI oppure DIRETTA.

STRUMENTI = il messaggio chiede o continua a parlare di un'informazione di viaggio, anche in
forma breve, una volere o desiderio di informazioni di viaggio o una richiesta di aiuto per pianificare un viaggio, controllare il meteo o cercare voli.
DIRETTA = il messaggio NON contiene nessuna richiesta di informazioni ma contiene: saluti, ringraziamenti,
congedi, commenti, domande sull'assistente, richieste di riformulare o riassumere.

Esempi:
"ciao" -> DIRETTA
"buongiorno, come stai?" -> DIRETTA
"grazie mille!" -> DIRETTA
"cosa sai fare?" -> DIRETTA
"puoi riassumere quello che mi hai detto?" -> DIRETTA
"cosa vedere a Roma?" -> STRUMENTI
"e a Lisbona?" -> STRUMENTI
"che tempo fa domani a Parigi?" -> STRUMENTI
"3 giorni, in due" -> STRUMENTI
"posso cancellare la prenotazione?" -> STRUMENTI"""


def _transcript(messages: list, last_n: int = 4, max_chars: int = 300) -> str:
    """Contesto recente + ultimo messaggio dell'utente, separati in modo netto."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(("Utente", str(m.content)[:max_chars]))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(("Assistente", str(m.content)[:max_chars]))
    if not lines:
        return ""

    last_role, last_text = lines[-1]
    context = "\n".join(f"{role}: {text}" for role, text in lines[-last_n:-1])
    parts = []
    if context:
        parts.append(f"Conversazione precedente:\n{context}")
    parts.append(f"Ultimo messaggio ({last_role.lower()}): {last_text}")
    parts.append("Classificazione dell'ultimo messaggio:")
    return "\n\n".join(parts)


def needs_search(llm, messages: list) -> bool:
    """True se servono gli strumenti (o se non si riesce a capirlo), False se si può rispondere subito."""
    try:
        reply = llm.invoke(
            [("system", ROUTER_PROMPT), ("human", _transcript(messages))],
            options={"num_predict": 8, "temperature": 0},
        )
        text = str(reply.content).upper()
    except Exception:
        logger.warning("Router non riuscito: si procede con gli strumenti.", exc_info=True)
        return True
    direct = "DIRETTA" in text and "STRUMENTI" not in text
    logger.info("router -> %s (%r)", "DIRETTA" if direct else "STRUMENTI", text.strip())
    return not direct