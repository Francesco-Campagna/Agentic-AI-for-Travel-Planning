"""Controllo di qualità del retrieval: per ogni domanda verifica che il file
giusto compaia tra i risultati.   Uso: python -m app.rag.check
"""

from .retriever import search_knowledge

# (domanda, file che dovrebbe comparire tra i risultati)
CASES = [
    ("Quando è chiuso il Louvre?", "parigi.md"),
    ("Quali sono i quartieri di Lisbona?", "lisbona.md"),
    ("Come si prenotano i biglietti per la casa di Anna Frank?", "amsterdam.md"),
    ("Cosa vedere a Barcellona con dei bambini?", "barcellona.md"),
    ("Itinerario di 3 giorni a Roma", "roma.md"),
    ("Posso cancellare la prenotazione 10 giorni prima della partenza?", "policy_agenzia.md"),
    ("Il volo ha 4 ore di ritardo, ho diritto a un risarcimento?", "fly_policy_and_normative.md"),
    ("Che misure ha il bagaglio a mano di Ryanair?", "fly_policy_and_normative.md"),
    ("Consigli per un viaggio low cost da solo", "profili_di_viaggio.md"),
    ("Come vado dall'aeroporto al centro di Amsterdam?", "amsterdam.md"),
]


def main() -> None:
    ok = 0
    for question, expected in CASES:
        results = search_knowledge(question)
        sources = [doc.metadata["source"] for doc, _ in results]
        found = any(s.endswith(expected) for s in sources)
        ok += found
        print(f"\n{'OK  ' if found else 'MISS'} {question}   (atteso: {expected})")
        for doc, score in results:
            print(f"      {score:.3f}  {doc.metadata['source']} › {doc.metadata['sezione']}")
    print(f"\nRisultato: {ok}/{len(CASES)} domande con il file giusto tra i primi risultati")


if __name__ == "__main__":
    main()