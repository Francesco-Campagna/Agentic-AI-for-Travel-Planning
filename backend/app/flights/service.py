"""Ricerca voli: dalla richiesta dell'utente al testo da restituire al modello.

    use:
        python -m app.flights.service Roma Parigi 2026-11-15 1          # cerca voli Roma → Parigi il 15 novembre 2026, 1 giorno
        python -m app.flights.service Roma Parigi 2026-11-15 3          # cerca voli Roma → Parigi dal 15 novembre 2026 per 3 giorni
"""

import logging

from datetime import date, datetime, timedelta

from . import db

logger = logging.getLogger("travel-agent")

GIORNI_SETTIMANA = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MAX_DAYS_SHOWN = 7

CITTA_IN_INGLESE = {
    "Roma": "Rome",
    "Parigi": "Paris",
    "Lisbona": "Lisbon",
    "Amsterdam": "Amsterdam",
    "Barcellona": "Barcelona",
}


class FlightsError(Exception):
    """Errore "previsto" (città non supportata, nessun volo...). Il messaggio è scritto per il modello e l'utente."""


def _elenco_citta() -> str:
    citta = db.citta_supportate()
    return ", ".join(citta[:-1]) + f" e {citta[-1]}"


def _risolvi(testo: str | None, ruolo: str) -> str:
    """Nome ufficiale della città o FlightsError. `ruolo` è "partenza" o "arrivo"."""
    if not testo or testo.strip().lower() in {"", "none", "null"}:
        raise FlightsError(f"Per cercare un volo mi serve la città di {ruolo}: chiedila all'utente.")
    # facciamo un check se la città è stata scritta in inglese e la convertiamo in italiano
    for it, en in CITTA_IN_INGLESE.items():
        if testo.strip().lower() == en.lower():
            testo = it
            break
    citta = db.trova_citta(testo)
    if citta is None:
        raise FlightsError(
            f"Non ho informazioni sui voli da o per '{testo.strip()}': posso cercare voli solo tra "
            f"{_elenco_citta()}. Riferiscilo all'utente e proponi, se ha senso, una di queste città. Ma non inventare altro."
        )
    return citta


def _durata(minuti: int) -> str:
    return f"{minuti // 60}h{minuti % 60:02d}"


def _gia_partito(volo, adesso: datetime) -> bool:
    """True se il volo di OGGI è già decollato. Gli orari sono locali: li riportiamo all'ora italiana."""
    off = db.aeroporto(volo["origine"])["offset_cet_h"]
    partenza = datetime.fromisoformat(f"{volo['data']}T{volo['ora_partenza']}") - timedelta(hours=off)
    return partenza <= adesso


def _parse_data(testo: str) -> date:
    testo = testo.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(testo, fmt).date()
        except ValueError:
            pass
    raise FlightsError(
        f"Non riesco a leggere la data '{testo}': serve il formato AAAA-MM-GG. "
        "Chiedi all'utente di indicarla meglio."
    )


def cerca(partenza: str | None, arrivo: str | None, data: str | None = None,
          giorni: int | None = None) -> tuple[str, dict]:
    """Voli diretti da `partenza` ad `arrivo`.

    data:   giorno di partenza (AAAA-MM-GG). Se assente si parte da oggi.
    giorni: quanti giorni consecutivi mostrare (1-7). Default: 1 se c'è `data`, 3 altrimenti.

    Restituisce (testo per il modello, fonte per l'interfaccia). Solleva FlightsError se non si può rispondere.
    """
    origine = _risolvi(partenza, "partenza")
    destinazione = _risolvi(arrivo, "arrivo")
    if origine == destinazione:
        raise FlightsError(f"Partenza e arrivo coincidono ({origine}): servono due città diverse.")

    oggi, adesso = db.oggi_e_adesso()
    if data and str(data).strip().lower() not in {"none", "null"}:
        dal = _parse_data(str(data))
        giorni = 1 if giorni is None else giorni
    else:
        dal = oggi
        giorni = 3 if giorni is None else giorni
    giorni = max(1, min(int(giorni), MAX_DAYS_SHOWN))

    if dal < oggi:
        raise FlightsError(
            f"Il {dal:%d/%m/%Y} è già passato (oggi è {GIORNI_SETTIMANA[oggi.weekday()]} {oggi:%d/%m/%Y}). "
            "Verifica con l'utente la data e l'anno."
        )
    al = dal + timedelta(days=giorni - 1)
    logger.info("flights: %s -> %s dal %s al %s", origine, destinazione, dal, al)

    primo, ultimo = db.copertura()
    if dal > ultimo:
        raise FlightsError(
            f"Posso mostrare voli solo fino al {ultimo:%d/%m/%Y}: il {dal:%d/%m/%Y} è oltre il periodo "
            "disponibile. Riferiscilo all'utente e proponi una data precedente."
        )

    voli = db.cerca_voli(origine, destinazione, dal, al)
    voli = [v for v in voli if not (v["data"] == oggi.isoformat() and _gia_partito(v, adesso))]
    if not voli:
        periodo = f"il {dal:%d/%m/%Y}" if dal == al else f"tra il {dal:%d/%m/%Y} e il {al:%d/%m/%Y}"
        raise FlightsError(
            f"Nessun volo da {origine} a {destinazione} {periodo}. "
            f"I voli in archivio vanno dal {primo:%d/%m/%Y} al {ultimo:%d/%m/%Y}."
        )

    ap_o, ap_d = db.aeroporto(origine), db.aeroporto(destinazione)
    righe = [
        f"Voli diretti da {origine} ({ap_o['codice']}) a {destinazione} ({ap_d['codice']}). Orari locali."
    ]
    for v in voli:
        giorno = datetime.fromisoformat(v["data"])
        posti = f"{v['posti_disponibili']} posti" if v["posti_disponibili"] > 0 else "esaurito"
        righe.append(
            f"- {GIORNI_SETTIMANA[giorno.weekday()]} {giorno:%d/%m/%Y}: {v['numero_volo']} ({v['compagnia']}), "
            f"partenza {v['ora_partenza']} {v['aeroporto_partenza']}, arrivo {v['ora_arrivo']} {v['aeroporto_arrivo']}, "
            f"durata {_durata(v['durata_min'])}, {v['prezzo_eur']:.0f} €, {posti}."
        )

    periodo = f"{dal:%d/%m/%Y}" if dal == al else f"{dal:%d/%m/%Y} – {al:%d/%m/%Y}"
    source = {"title": f"Voli {origine} → {destinazione} › {periodo}", "source": "Database voli"}
    return "\n".join(righe), source

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    a = sys.argv[1:] + [None] * 4   # Roma Parigi 2026-11-15 1
    try:
        text, src = cerca(a[0], a[1], a[2], int(a[3]) if a[3] else None)
        print("\n" + text)
        print("\nFonte per il frontend:", src)
    except FlightsError as err:
        print("\nErrore:", err)