"""Strumenti (tool) che l'agente può decidere di invocare."""

import logging

from langchain_core.tools import tool
from ..config import settings

from ..rag.retriever import search_knowledge
from .weather import WeatherError, get_weather
from ..flights.service import FlightsError, cerca as cerca_voli_db

logger = logging.getLogger("travel-agent")

# Etichette mostrate al modello: servono a distinguere la policy dell'agenzia dalle regole reali.
DOC_TYPE_LABELS = {
    "destination_guide": "Guida alla destinazione",
    "agency_policy": "Policy dell'agenzia",
    "external_policies": "Normativa e regole reali (UE, compagnie aeree)",
    "traveler_profiles": "Profili di viaggio",
}

def _empty(value: str | None) -> bool:
    # I modelli piccoli a volte scrivono "None" o "null" come testo invece di omettere il campo.
    return value is None or value.strip().lower() in {"", "none", "null"}

@tool(parse_docstring=True, response_format="content_and_artifact")
def cerca_conoscenza(query: str, destinazione: str | None = None) -> tuple[str, list[dict]]:
    """Cerca nella knowledge base dell'agenzia di viaggio: guide delle destinazioni (quartieri, attrazioni e musei con orari e giorni di chiusura, cucina, trasporti, itinerari), profili di viaggio, policy dell'agenzia (prenotazioni, cancellazioni, rimborsi, assicurazione, bagagli) e regole reali su voli e diritti dei passeggeri.

    Args:
        query: Cosa cercare: l'argomento e, se c'è, la città. Frase completa e autonoma, senza riferimenti ai messaggi precedenti. Se la domanda è un seguito come "e a Lisbona?", riprendi l'argomento precedente con il nuovo luogo.
        destinazione: Nome della città a cui limitare la ricerca. Lasciala vuota per policy, bagagli, voli e domande non legate a una città.
    """
    if _empty(destinazione):
        destinazione = None

    results = search_knowledge(query, destinazione)
    if not results and destinazione:  # filtro troppo stretto (grafia diversa dai metadati): riprova senza
        logger.info("cerca_conoscenza: nessun risultato con destinazione=%r, riprovo senza filtro", destinazione)
        results = search_knowledge(query, None)

    if results:  # tieni solo i pezzi vicini al migliore: gli altri sono quasi sempre rumore
        best = results[0][1]
        results = [(doc, score) for doc, score in results if score >= best - settings.rag_score_margin]
    best = f"{results[0][1]:.3f}" if results else "-"
    logger.info("cerca_conoscenza(query=%r, destinazione=%r) -> %d risultati, miglior punteggio %s",
                query, destinazione, len(results), best)
    if not results:
        return "Nessun risultato trovato nella knowledge base.", []

    blocks: list[str] = []
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for i, (doc, _score) in enumerate(results, start=1):
        label = DOC_TYPE_LABELS.get(doc.metadata.get("doc_type"), "Documento")
        blocks.append(f"[{i}] ({label})\n{doc.page_content}")

        key = (doc.metadata["source"], doc.metadata["sezione"])
        if key not in seen:
            seen.add(key)
            sources.append({"title": f"{doc.metadata['titolo']} › {doc.metadata['sezione']}", "source": doc.metadata["source"]})

    # Primo elemento: testo per il modello. Secondo: dati strutturati per l'interfaccia (non vanno al modello).
    return "\n\n".join(blocks), sources


@tool(parse_docstring=True, response_format="content_and_artifact")
def meteo_citta(citta: str, tra_giorni: int = 0, giorni: int = 1) -> tuple[str, list[dict]]:
    """Meteo REALE di una città: condizioni attuali e previsioni fino a 15 giorni da oggi. Usalo SEMPRE per "che tempo fa", "domani piove?", "serve l'ombrello?", "previsioni per il weekend". Non usarlo per il clima in generale o per il periodo migliore in cui viaggiare: per quello c'è cerca_conoscenza.

    Args:
        citta: Nome di UNA sola città, per esempio "Roma" o "Lisbona".
        tra_giorni: Tra quanti giorni da oggi. 0 = oggi (o se l'utente non indica il giorno), 1 = domani, 2 = dopodomani, 3 = "tra 3 giorni", e così via. Non conosci la data di oggi: non inventare date. Se l'utente indica una data precisa lontana e non sai quanti giorni mancano, chiedilo.
        giorni: Quanti giorni consecutivi mostrare a partire da tra_giorni. 1 per un giorno solo, da 2 a 7 per "previsioni" o "prossimi giorni". Se l'utente indica un giorno della settimana o una data vicina (es. "sabato", "il 30 ottobre") usa tra_giorni=0 e giorni=7, poi scegli nella risposta il giorno richiesto.
    """
    logger.info("meteo_citta(citta=%r, tra_giorni=%r, giorni=%r) chiamato dal modello", citta, tra_giorni, giorni)

    if _empty(citta):
        return "Per darti il meteo mi serve il nome della città.", []

    try:
        text, source = get_weather(citta.strip(), tra_giorni, giorni)
    except WeatherError as exc:
        # Errore "normale": lo restituiamo come testo, così il modello lo spiega all'utente.
        logger.info("meteo_citta -> errore gestito: %s", exc)
        return str(exc), []

    logger.info("meteo_citta -> ok (%d caratteri restituiti al modello)", len(text))
    return text, [source]


@tool(parse_docstring=True, response_format="content_and_artifact")
def cerca_voli(partenza: str, arrivo: str, data: str | None = None,
               giorni: int | None = None) -> tuple[str, list[dict]]:
    """Cerca voli diretti con orari, compagnia, durata, prezzo e posti disponibili. Funziona SOLO tra Roma, Parigi, Lisbona, Amsterdam e Barcellona: per qualsiasi altra città lo strumento risponde che non ha informazioni, quindi usalo comunque per ogni domanda su voli, orari di volo, durata del volo o prezzi dei biglietti aerei. Non usarlo per bagagli, ritardi, rimborsi o diritti dei passeggeri: per quello c'è cerca_conoscenza. Per andata e ritorno fai due chiamate (una per tratta).

    Args:
        partenza: Città di partenza, per esempio "Roma". Una sola città.
        arrivo: Città di arrivo, per esempio "Lisbona". Una sola città.
        data: Giorno di partenza in formato AAAA-MM-GG, per esempio "2026-11-15". Calcolalo sempre a partire dalla data di oggi indicata nel prompt: "domani", "venerdì", "il 15 novembre", "tra 10 giorni" vanno tutti convertiti in una data esplicita. Se l'utente non indica l'anno, usa la prossima occorrenza di quel giorno. Ometti il parametro solo se l'utente non indica nessun giorno.
        giorni: Quanti giorni consecutivi mostrare a partire da data. Di solito omettilo (1 se hai indicato la data, 3 altrimenti). Usa 7 solo per richieste generiche come "la prossima settimana" o "nei prossimi giorni".
    """
    logger.info("cerca_voli(partenza=%r, arrivo=%r, data=%r, giorni=%r) chiamato dal modello",
                partenza, arrivo, data, giorni)
    try:
        text, source = cerca_voli_db(partenza, arrivo, data, giorni)
    except FlightsError as exc:
        logger.info("cerca_voli -> risposta senza voli: %s", exc)
        return str(exc), []

    logger.info("cerca_voli -> ok (%d caratteri restituiti al modello)", len(text))
    return text, [source]