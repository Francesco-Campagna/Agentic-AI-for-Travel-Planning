"""Dati di partenza del database voli (SIMULATI, a scopo dimostrativo).

Per ogni coppia di città (in entrambe le direzioni = 20 tratte) vengono creati 3 voli
al giorno, con orari e numeri di volo fissi, cambiano da un giorno all'altro 
solo prezzo e posti disponibili. La generazione è deterministica:
rigenerando il database si ottengono gli stessi voli.
"""

import logging
import random
import sqlite3
from datetime import date, datetime, timedelta
from itertools import combinations

logger = logging.getLogger("travel-agent")

# citta -> (codice IATA, nome aeroporto, ore di differenza rispetto all'ora dell'Europa centrale)
AEROPORTI = {
    "Roma": ("FCO", "Roma Fiumicino", 0),
    "Parigi": ("CDG", "Parigi Charles de Gaulle", 0),
    "Lisbona": ("LIS", "Lisbona Humberto Delgado", -1),
    "Amsterdam": ("AMS", "Amsterdam Schiphol", 0),
    "Barcellona": ("BCN", "Barcellona El Prat", 0),
}

# (città A, città B) -> (durata in minuti, prezzo base in euro, compagnie [(codice, nome), ...])
# Vale per entrambe le direzioni.
ROTTE = {
    ("Roma", "Parigi"): (125, 120, [("AZ", "ITA Airways"), ("AF", "Air France")]),
    ("Roma", "Lisbona"): (185, 140, [("TP", "TAP Air Portugal"), ("AZ", "ITA Airways")]),
    ("Roma", "Amsterdam"): (140, 130, [("KL", "KLM"), ("AZ", "ITA Airways")]),
    ("Roma", "Barcellona"): (110, 90, [("VY", "Vueling"), ("AZ", "ITA Airways")]),
    ("Parigi", "Lisbona"): (150, 120, [("TP", "TAP Air Portugal"), ("AF", "Air France")]),
    ("Parigi", "Amsterdam"): (75, 90, [("KL", "KLM"), ("AF", "Air France")]),
    ("Parigi", "Barcellona"): (115, 100, [("VY", "Vueling"), ("AF", "Air France")]),
    ("Lisbona", "Amsterdam"): (185, 140, [("KL", "KLM"), ("TP", "TAP Air Portugal")]),
    ("Lisbona", "Barcellona"): (125, 90, [("VY", "Vueling"), ("TP", "TAP Air Portugal")]),
    ("Amsterdam", "Barcellona"): (145, 110, [("VY", "Vueling"), ("KL", "KLM")]),
}

# Fasce orarie di partenza dei 3 voli giornalieri: (ora minima, ora massima)
FASCE_ORARIE = [(6, 9), (11, 14), (16, 20)]
# Moltiplicatore di prezzo per fascia (mattina presto e sera tardi costano meno/più)
FATTORE_FASCIA = [0.95, 1.0, 1.1]


def _rotta(origine: str, destinazione: str) -> tuple[int, int, list[tuple[str, str]]]:
    return ROTTE.get((origine, destinazione)) or ROTTE[(destinazione, origine)]


def _orario_di_linea(origine: str, destinazione: str) -> list[dict]:
    """I 3 voli giornalieri della tratta: stessi orari e numeri di volo ogni giorno."""
    durata, _, compagnie = _rotta(origine, destinazione)
    rng = random.Random(f"orario-{origine}-{destinazione}")
    voli = []
    for i, (h_min, h_max) in enumerate(FASCE_ORARIE):
        codice, nome = compagnie[i % len(compagnie)]
        minuti_partenza = rng.randrange(h_min * 60, h_max * 60, 5)
        voli.append({
            "numero_volo": f"{codice}{rng.randint(100, 999)}",
            "compagnia": nome,
            "minuti_partenza": minuti_partenza,
            "durata_min": durata + rng.choice([-5, 0, 0, 5]),
            "fattore": FATTORE_FASCIA[i],
        })
    return voli


def _hhmm(minuti: int) -> str:
    return f"{minuti // 60 % 24:02d}:{minuti % 60:02d}"


def genera_voli(da: date, giorni: int) -> list[tuple]:
    """Tutte le righe della tabella `voli` per `giorni` giorni a partire da `da`."""
    righe = []
    for a, b in combinations(AEROPORTI, 2):
        for origine, destinazione in ((a, b), (b, a)):
            cod_o, _, off_o = AEROPORTI[origine]
            cod_d, _, off_d = AEROPORTI[destinazione]
            _, prezzo_base, _ = _rotta(origine, destinazione)
            orario = _orario_di_linea(origine, destinazione)
            for n in range(giorni):
                giorno = da + timedelta(days=n)
                for volo in orario:
                    rng = random.Random(f"{volo['numero_volo']}-{giorno.isoformat()}")
                    # più caro nel weekend e a ridosso della partenza
                    fattore = volo["fattore"] * (1.15 if giorno.weekday() in (4, 6) else 1.0)
                    fattore *= 1.25 if n < 3 else (1.1 if n < 10 else 1.0)
                    prezzo = round(prezzo_base * fattore * rng.uniform(0.85, 1.2))
                    arrivo = volo["minuti_partenza"] + volo["durata_min"] + (off_d - off_o) * 60
                    righe.append((
                        volo["numero_volo"], volo["compagnia"],
                        origine, destinazione, cod_o, cod_d,
                        giorno.isoformat(), _hhmm(volo["minuti_partenza"]), _hhmm(arrivo),
                        volo["durata_min"], float(prezzo), rng.randint(0, 70),
                    ))
    return righe


def popola(conn: sqlite3.Connection, da: date, giorni: int) -> int:
    """Svuota e riempie `aeroporti` e `voli`. Restituisce il numero di voli creati."""
    conn.execute("DELETE FROM voli")
    conn.execute("DELETE FROM aeroporti")
    conn.executemany(
        "INSERT INTO aeroporti (citta, codice, nome, offset_cet_h) VALUES (?, ?, ?, ?)",
        [(citta, cod, nome, off) for citta, (cod, nome, off) in AEROPORTI.items()],
    )
    righe = genera_voli(da, giorni)
    conn.executemany(
        """INSERT INTO voli (numero_volo, compagnia, origine, destinazione, aeroporto_partenza,
               aeroporto_arrivo, data, ora_partenza, ora_arrivo, durata_min, prezzo_eur, posti_disponibili)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        righe,
    )
    conn.commit()
    logger.info("flights: database popolato con %d voli dal %s per %d giorni", len(righe), da, giorni)
    return len(righe)


if __name__ == "__main__":
    # Rigenera il database da terminale: python -m app.flights.seed
    from .db import rebuild

    logging.basicConfig(level=logging.INFO)
    rebuild()