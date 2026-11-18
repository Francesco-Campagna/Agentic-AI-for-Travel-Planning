"""Database voli (SQLite): schema, connessione e query.

Il file viene creato e popolato all'avvio del backend (con 'ensure_ready').

Tabelle:
  aeroporti(citta, codice, nome, offset_cet_h)
  voli(id, numero_volo, compagnia, origine, destinazione, aeroporto_partenza, aeroporto_arrivo,
       data, ora_partenza, ora_arrivo, durata_min, prezzo_eur, posti_disponibili)
Date in formato AAAA-MM-GG, orari HH:MM nell'ora LOCALE dell'aeroporto.
"""

import logging
import sqlite3
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from ..config import settings
from . import seed

logger = logging.getLogger("travel-agent")

SCHEMA = """
CREATE TABLE IF NOT EXISTS aeroporti (
    citta         TEXT PRIMARY KEY,
    codice        TEXT NOT NULL,
    nome          TEXT NOT NULL,
    offset_cet_h  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS voli (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    numero_volo         TEXT NOT NULL,
    compagnia           TEXT NOT NULL,
    origine             TEXT NOT NULL,
    destinazione        TEXT NOT NULL,
    aeroporto_partenza  TEXT NOT NULL,
    aeroporto_arrivo    TEXT NOT NULL,
    data                TEXT NOT NULL,
    ora_partenza        TEXT NOT NULL,
    ora_arrivo          TEXT NOT NULL,
    durata_min          INTEGER NOT NULL,
    prezzo_eur          REAL NOT NULL,
    posti_disponibili   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voli_tratta ON voli (origine, destinazione, data);
"""


def connect() -> sqlite3.Connection:
    path = Path(settings.flights_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def oggi_e_adesso() -> tuple[date, datetime]:
    """Data e ora di adesso nel fuso orario configurato"""
    adesso = datetime.now(ZoneInfo(settings.flights_timezone))
    return adesso.date(), adesso.replace(tzinfo=None)


def rebuild() -> int:
    """Ricrea il contenuto del database a partire da oggi."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        return seed.popola(conn, oggi_e_adesso()[0], settings.flights_days_ahead)


@lru_cache(maxsize=1)
def ensure_ready() -> None:
    """Crea lo schema e (ri)popola i voli se mancano o stanno per finire. Eseguita una volta per processo."""
    oggi, _ = oggi_e_adesso()
    with connect() as conn:
        conn.executescript(SCHEMA)
        ultimo = conn.execute("SELECT MAX(data) FROM voli").fetchone()[0]
    if ultimo is None or date.fromisoformat(ultimo) < oggi + timedelta(days=settings.flights_min_coverage_days):
        logger.info("flights: voli mancanti o in esaurimento (ultimo giorno: %s) -> rigenerazione", ultimo)
        rebuild()
    else:
        logger.info("flights: database pronto (voli fino al %s)", ultimo)


def _normalizza(testo: str) -> str:
    """Minuscolo e senza accenti/spazi esterni: "  Barcellòna " -> "barcellona"."""
    senza_accenti = unicodedata.normalize("NFKD", testo).encode("ascii", "ignore").decode()
    return senza_accenti.strip().lower()


def citta_supportate() -> list[str]:
    ensure_ready()
    with connect() as conn:
        return [r["citta"] for r in conn.execute("SELECT citta FROM aeroporti ORDER BY citta")]


def trova_citta(testo: str) -> str | None:
    """Nome ufficiale della città (o None). Accetta anche il codice dell'aeroporto: "fco" -> "Roma"."""
    ensure_ready()
    chiave = _normalizza(testo)
    with connect() as conn:
        for r in conn.execute("SELECT citta, codice FROM aeroporti"):
            if chiave in (_normalizza(r["citta"]), r["codice"].lower()):
                return r["citta"]
    return None


def aeroporto(citta: str) -> sqlite3.Row:
    with connect() as conn:
        return conn.execute("SELECT * FROM aeroporti WHERE citta = ?", (citta,)).fetchone()


def copertura() -> tuple[date, date]:
    ensure_ready()
    with connect() as conn:
        primo, ultimo = conn.execute("SELECT MIN(data), MAX(data) FROM voli").fetchone()
    return date.fromisoformat(primo), date.fromisoformat(ultimo)


def cerca_voli(origine: str, destinazione: str, dal: date, al: date) -> list[sqlite3.Row]:
    """Voli della tratta con data tra `dal` e `al` (estremi inclusi), in ordine di partenza."""
    ensure_ready()
    with connect() as conn:
        return conn.execute(
            """SELECT * FROM voli
               WHERE origine = ? AND destinazione = ? AND data BETWEEN ? AND ?
               ORDER BY data, ora_partenza""",
            (origine, destinazione, dal.isoformat(), al.isoformat()),
        ).fetchall()