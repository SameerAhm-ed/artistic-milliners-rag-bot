import { useState, useRef, useEffect } from "react";
import { sendMessage } from "./api";
import "./App.css";

const SESSION_ID = crypto.randomUUID();

const SUGGESTIONS = [
  "What is the MOQ?",
  "What are typical lead times?",
  "What fabrics do you offer?",
  "What's your returns policy?",
];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function submit(text) {
    if (!text || loading) return;
    setError(null);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await sendMessage(SESSION_ID, text);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.answer, sources: res.sources, chunks: res.chunks },
      ]);
    } catch (err) {
      setError("Could not reach the assistant. Is the backend running?");
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSend(e) {
    e.preventDefault();
    submit(input.trim());
  }

  return (
    <div className="app">
      <header className="header">
        <p className="header-eyebrow">Artistic Milliners</p>
        <h1 className="header-title">Support Assistant</h1>
      </header>

      <main className="chat" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty-title">Ask about sourcing, denim, or delivery.</p>
            <p className="empty-sub">
              Answers are grounded in our knowledge base and cite their sources.
            </p>
            <div className="empty-suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => submit(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} m={m} />
        ))}
        {loading && (
          <div className="msg assistant" aria-label="Assistant is typing">
            <div className="bubble bubble-loading">
              <TypingDots />
            </div>
          </div>
        )}
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        <div ref={endRef} />
      </main>

      <form className="composer" onSubmit={handleSend}>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question…"
          aria-label="Message"
          autoComplete="off"
        />
        <button type="submit" disabled={loading || !input.trim()} aria-label="Send message">
          <SendIcon />
        </button>
      </form>
    </div>
  );
}

function Message({ m }) {
  const [open, setOpen] = useState(false);
  const isUser = m.role === "user";
  return (
    <div className={`msg ${m.role}`}>
      <div className="bubble">{m.content}</div>
      {!isUser && m.sources?.length > 0 && (
        <div className="sources">
          <button
            type="button"
            className="sources-toggle"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
          >
            <span className="sources-toggle-caret" data-open={open}>
              ▸
            </span>
            {open ? "Hide sources" : `Sources (${m.sources.length})`}
          </button>
          {open && (
            <div className="sources-body">
              <div className="source-tags">
                {m.sources.map((s, i) => (
                  <span key={i} className="source-tag">
                    <span className="source-tag-file">{s.source_file}</span>
                    <span className="source-tag-sep">·</span>
                    <span className="source-tag-section">{s.section_title}</span>
                  </span>
                ))}
              </div>
              <details className="chunks">
                <summary>Retrieved passages</summary>
                {m.chunks.map((c, i) => (
                  <p key={i} className="chunk">
                    {c.text}
                  </p>
                ))}
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <span className="dots" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </span>
  );
}

function SendIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path
        d="M17.5 2.5L2.5 8.75L9.375 10.625M17.5 2.5L11.25 17.5L9.375 10.625M17.5 2.5L9.375 10.625"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
