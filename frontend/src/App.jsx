import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL ?? "/api/chat";
// Endpoint della cronologia, derivato da quello della chat (…/chat -> …/history)
const HISTORY_URL = API_URL.replace(/\/chat$/, "/history");
const MOCK = import.meta.env.VITE_MOCK === "true";

const SUGGESTIONS = [
  "Pianificami 3 giorni a Roma per una coppia",
  "Che tempo farà a Lisbona la prossima settimana?",
  "Cerca voli Roma–Barcellona domani",
  "Cosa prevede la vostra policy di cancellazione?",
];

// Un id per scheda del browser: il backend lo usa per la memoria della conversazione.
function getSessionId() {
  try {
    let id = sessionStorage.getItem("tpa_session");
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem("tpa_session", id);
    }
    return id;
  } catch {
    return crypto.randomUUID();
  }
}
const SESSION_ID = getSessionId();

// Contratto atteso dal backend:
//   POST /api/chat  { session_id, message }
//   -> { answer: string, sources?: [{ title, source }], tools?: [string] }
async function askAgent(message) {
  if (MOCK) {
    await new Promise((r) => setTimeout(r, 900));
    return {
      answer: `Risposta di prova a: **${message}**\n\n- Punto uno\n- Punto due`,
      sources: [{ title: "Roma – guida", source: "destinations/roma.md" }],
      tools: ["meteo"],
    };
  }
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID, message }),
  });
  if (!res.ok) throw new Error(`Il server ha risposto con errore ${res.status}`);
  return res.json();
}

// Recupera dal backend i messaggi già scambiati in questa sessione.
// Se fallisce (backend non raggiungibile, ecc.) si riparte da chat vuota.
async function loadHistory() {
  if (MOCK) return [];
  try {
    const res = await fetch(`${HISTORY_URL}/${encodeURIComponent(SESSION_ID)}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.messages ?? [];
  } catch {
    return [];
  }
}

function Message({ m }) {
  const isUser = m.role === "user";
  return (
    <div className={`row ${isUser ? "row-user" : "row-agent"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : m.error ? "bubble-error" : "bubble-agent"}`}>
        {isUser ? m.text : <ReactMarkdown>{m.text}</ReactMarkdown>}
        {m.tools?.length > 0 && (
          <p className="meta">Strumenti usati: {m.tools.join(", ")}</p>
        )}
        {m.sources?.length > 0 && (
          <details className="sources">
            <summary>Fonti ({m.sources.length})</summary>
            <ul>
              {m.sources.map((s, i) => (
                <li key={i}>{s.title}{s.source ? ` – ${s.source}` : ""}</li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyReady, setHistoryReady] = useState(false);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  // Al caricamento della pagina ripristina la conversazione della sessione.
  useEffect(() => {
    let cancelled = false;
    loadHistory().then((past) => {
      if (cancelled) return;
      // Se nel frattempo l'utente ha già scritto, non sovrascrivere i suoi messaggi.
      setMessages((prev) => (prev.length === 0 ? past : prev));
      setHistoryReady(true);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }, [input]);

  async function send(text) {
    const message = text.trim();
    if (!message || loading) return;
    setMessages((prev) => [...prev, { role: "user", text: message }]);
    setInput("");
    setLoading(true);
    try {
      const data = await askAgent(message);
      setMessages((prev) => [
        ...prev,
        { role: "agent", text: data.answer, sources: data.sources, tools: data.tools },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "agent", error: true, text: `Non riesco a contattare l'assistente. ${err.message}. Riprova tra poco.` },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="app">
      <header className="top">
        <h1>Travel Planning Agent</h1>
        <p>Il tuo assistente per pianificare il viaggio</p>
      </header>

      <main className="log" role="log" aria-live="polite">
        {historyReady && messages.length === 0 && (
          <section className="empty">
            <h2>Dove vuoi andare?</h2>
            <p>Chiedimi di destinazioni, itinerari, meteo, voli o regole di viaggio.</p>
            <div className="chips">
              {SUGGESTIONS.map((s) => (
                <button key={s} onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </section>
        )}
        {messages.map((m, i) => <Message key={i} m={m} />)}
        {loading && (
          <div className="row row-agent" aria-label="L'assistente sta scrivendo">
            <div className="bubble bubble-agent typing"><span /><span /><span /></div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Scrivi un messaggio…"
          aria-label="Messaggio"
        />
        <button type="submit" disabled={loading || !input.trim()}>Invia</button>
      </form>
    </div>
  );
}
