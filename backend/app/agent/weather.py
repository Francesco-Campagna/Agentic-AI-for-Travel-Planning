"""Client meteo basato su Open-Meteo (https://open-meteo.com).

1. geocoding: nome della città -> latitudine/longitudine
2. forecast:  coordinate -> meteo attuale + previsioni giornaliere

Esempi di chiamata da riga di comando (per testare il servizio senza passare dal modello):
    python -m app.agent.weather Roma         # oggi
    python -m app.agent.weather Roma 1       # domani
    python -m app.agent.weather Roma 2 3     # da dopodomani, per 3 giorni
    python -m app.agent.weather Roma 0 7     # i prossimi 7 giorni
"""

import logging
from datetime import date

import httpx

from ..config import settings

logger = logging.getLogger("travel-agent")

GIORNI_SETTIMANA = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MAX_FORECAST_DAYS = 16  # massimo offerto da Open-Meteo
MAX_DAYS_SHOWN = 7      # tetto ai giorni mostrati in una risposta

# Codici meteo WMO restituiti da Open-Meteo -> descrizione in italiano.
WMO_CODES = {
    0: "cielo sereno",
    1: "prevalentemente sereno",
    2: "parzialmente nuvoloso",
    3: "coperto",
    45: "nebbia",
    48: "nebbia con brina",
    51: "pioggerella leggera",
    53: "pioggerella",
    55: "pioggerella intensa",
    56: "pioggerella gelata leggera",
    57: "pioggerella gelata intensa",
    61: "pioggia debole",
    63: "pioggia moderata",
    65: "pioggia forte",
    66: "pioggia gelata leggera",
    67: "pioggia gelata forte",
    71: "neve debole",
    73: "neve moderata",
    75: "neve forte",
    77: "granelli di neve",
    80: "rovesci deboli",
    81: "rovesci moderati",
    82: "rovesci violenti",
    85: "rovesci di neve deboli",
    86: "rovesci di neve forti",
    95: "temporale",
    96: "temporale con grandine leggera",
    99: "temporale con grandine forte",
}


class WeatherError(Exception):
    """Errore "previsto" (città non trovata, servizio non raggiungibile...).

    Il messaggio è scritto per essere letto dal modello e riferito all'utente.
    """


def _get_json(url: str, params: dict) -> dict:
    """GET con timeout e gestione errori. Ogni chiamata viene registrata nei log."""
    logger.info("meteo: GET %s %s", url, params)
    try:
        resp = httpx.get(url, params=params, timeout=settings.weather_timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException as exc:
        logger.warning("meteo: timeout su %s", url)
        raise WeatherError("Il servizio meteo non ha risposto in tempo. Riprova tra poco.") from exc
    except httpx.HTTPError as exc:
        logger.warning("meteo: errore HTTP su %s: %s", url, exc)
        raise WeatherError("Il servizio meteo non è al momento raggiungibile.") from exc


def geocode(city: str) -> dict:
    """Nome città -> {name, latitude, longitude, country, admin1}. Prende il risultato più popoloso."""
    data = _get_json(
        settings.weather_geocoding_url,
        {"name": city, "count": 1, "language": "it", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        logger.info("meteo: geocoding senza risultati per %r", city)
        raise WeatherError(f"Non ho trovato nessuna città chiamata '{city}'.")
    place = results[0]
    logger.info(
        "meteo: geocoding %r -> %s, %s (%s, %s)",
        city, place.get("name"), place.get("country"), place.get("latitude"), place.get("longitude"),
    )
    return place


def _describe(code) -> str:
    return WMO_CODES.get(code, f"condizioni non classificate (codice {code})")


def _label(d: date, today: date | None = None) -> str:
    label = f"{GIORNI_SETTIMANA[d.weekday()]} {d.day:02d}/{d.month:02d}/{d.year}"
    return f"{label} (oggi)" if d == today else label


def _v(value, unit: str = "") -> str:
    """Valore o "n.d." (Open-Meteo restituisce null per alcuni dati lontani nel tempo)."""
    return "n.d." if value is None else f"{value}{unit}"


def get_weather(city: str, offset: int = 0, days: int = 1) -> tuple[str, dict]:
    """Meteo di una città.

    offset: tra quanti giorni da oggi si parte (0 = oggi, 1 = domani, ...)
    days:   quanti giorni consecutivi mostrare (1-7)

    Il meteo ATTUALE è incluso solo se si parte da oggi.
    Restituisce (testo per il modello, fonte per l'interfaccia). Solleva WeatherError se qualcosa va storto.
    """
    offset = max(0, int(offset))
    days = max(1, min(int(days), MAX_DAYS_SHOWN))

    if offset >= MAX_FORECAST_DAYS:
        raise WeatherError(f"Posso dare previsioni al massimo fino a {MAX_FORECAST_DAYS - 1} giorni da oggi.")

    place = geocode(city)

    # "Oggi" è quello della città (timezone=auto): non dipende dall'orologio del server.
    params = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": min(offset + days, MAX_FORECAST_DAYS),
    }
    if offset == 0:
        params["current"] = "temperature_2m,apparent_temperature,weather_code"

    data = _get_json(settings.weather_forecast_url, params)

    cur = data.get("current") or {}
    daily = data.get("daily") or {}
    times = [date.fromisoformat(t) for t in daily.get("time") or []]
    if len(times) <= offset:
        raise WeatherError("Il servizio meteo ha restituito dati incompleti.")

    selected = list(range(offset, len(times)))
    logger.info("meteo: tra_giorni=%d giorni=%d -> %s", offset, days, [times[i].isoformat() for i in selected])

    where = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    lines = [f"Meteo per {where} (fonte: Open-Meteo)."]

    if cur:
        lines.append(
            f"Adesso: {_v(cur.get('temperature_2m'), '°C')} "
            f"(percepiti {_v(cur.get('apparent_temperature'), '°C')}), "
            f"{_describe(cur.get('weather_code'))}."
        )

    lines.append("Previsioni:" if len(selected) > 1 else "Previsione:")
    for i in selected:
        lines.append(
            f"- {_label(times[i], times[0])}: {_describe(daily['weather_code'][i])}, "
            f"minima {_v(daily['temperature_2m_min'][i], '°C')}, massima {_v(daily['temperature_2m_max'][i], '°C')}, "
            f"probabilità di pioggia {_v(daily['precipitation_probability_max'][i], '%')}."
        )

    first, last = times[selected[0]], times[selected[-1]]
    period = _label(first) if len(selected) == 1 else f"{_label(first)} – {_label(last)}"
    source = {"title": f"Meteo {place.get('name')} › {period}", "source": "https://open-meteo.com"}
    return "\n".join(lines), source


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    city_arg = sys.argv[1] if len(sys.argv) > 1 else "Roma"
    offset_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    days_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    try:
        text, src = get_weather(city_arg, offset_arg, days_arg)
        print("\n" + text)
        print("\nFonte per il frontend:", src)
    except WeatherError as err:
        print("\nErrore:", err)