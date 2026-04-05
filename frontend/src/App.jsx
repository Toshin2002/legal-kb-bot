import { useState, useRef, useEffect } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000";

const SUGGESTIONS = [
  "What is a contract?",
  "Can my landlord evict me without notice?",
  "What are my rights as an employee?",
  "What is small claims court?",
  "What does copyright protect?",
  "What is at-will employment?",
];

// Generate a simple session ID so the backend can track last retrieved chunks
const SESSION_ID = "session_" + Math.random().toString(36).slice(2, 9);

// ── Subcomponents ──────────────────────────────────────────────────────────────

function SourceTag({ name }) {
  return (
    <span style={{
      display: "inline-block",
      background: "#eef2ff", color: "#3730a3",
      border: "1px solid #c7d2fe", borderRadius: "99px",
      fontSize: "11px", padding: "2px 10px",
      marginRight: "6px", marginTop: "6px",
    }}>{name}</span>
  );
}

function CorrectionBadge() {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "4px",
      background: "#fef3c7", color: "#92400e",
      border: "1px solid #fde68a", borderRadius: "99px",
      fontSize: "11px", padding: "2px 10px",
      marginTop: "6px",
    }}>
      ✎ Correction applied
    </span>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  const isCorrectCmd = isUser && msg.content.toLowerCase().startsWith("/correct");

  return (
    <div style={{
      display: "flex",
      flexDirection: isUser ? "row-reverse" : "row",
      gap: "10px", alignItems: "flex-start",
      maxWidth: "800px", width: "100%",
      alignSelf: isUser ? "flex-end" : "flex-start",
    }}>
      <div style={{
        width: "32px", height: "32px", borderRadius: "50%",
        background: isUser ? "#4f46e5" : "#1e1b4b",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: "13px", fontWeight: "600", color: "white", flexShrink: 0,
      }}>
        {isUser ? "U" : "⚖"}
      </div>

      <div style={{
        background: isUser
          ? (isCorrectCmd ? "#fef3c7" : "#4f46e5")
          : "white",
        color: isUser
          ? (isCorrectCmd ? "#92400e" : "white")
          : "#111827",
        borderRadius: "16px",
        borderTopRightRadius: isUser ? "4px" : "16px",
        borderTopLeftRadius: isUser ? "16px" : "4px",
        padding: "12px 16px", fontSize: "14px", lineHeight: "1.7",
        border: isUser
          ? (isCorrectCmd ? "1px solid #fde68a" : "none")
          : "1px solid #e5e7eb",
        maxWidth: "calc(100% - 44px)",
      }}>
        <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>

        {/* Correction applied badge */}
        {msg.correctionApplied && (
          <div style={{ marginTop: "8px" }}>
            <CorrectionBadge />
          </div>
        )}

        {/* Source tags */}
        {msg.sources?.length > 0 && (
          <div style={{ marginTop: "10px", paddingTop: "8px", borderTop: "1px solid #e5e7eb" }}>
            <div style={{ fontSize: "11px", color: "#6b7280", marginBottom: "2px" }}>Sources</div>
            {msg.sources.map((s, i) => <SourceTag key={i} name={s} />)}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: "10px", alignItems: "flex-start" }}>
      <div style={{
        width: "32px", height: "32px", borderRadius: "50%",
        background: "#1e1b4b", display: "flex", alignItems: "center",
        justifyContent: "center", fontSize: "14px", flexShrink: 0,
      }}>⚖</div>
      <div style={{
        background: "white", border: "1px solid #e5e7eb",
        borderRadius: "16px", borderTopLeftRadius: "4px",
        padding: "14px 18px", display: "flex", gap: "5px", alignItems: "center",
      }}>
        {[0, 200, 400].map((delay, i) => (
          <div key={i} style={{
            width: "7px", height: "7px", borderRadius: "50%",
            background: "#9ca3af",
            animation: `bounce 1.2s ${delay}ms infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

function CommandHint({ visible }) {
  if (!visible) return null;
  return (
    <div style={{
      padding: "8px 20px",
      background: "#fefce8",
      borderTop: "1px solid #fde68a",
      fontSize: "12px", color: "#92400e",
    }}>
      <strong>/correct</strong> — type your corrected answer after this command to override the last bot response for similar questions.
      &nbsp;Example: <code>/correct A contract must always have consideration.</code>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState([{
    role: "assistant",
    content: "Hello! I'm LegalBot, your AI legal assistant.\n\nAsk me anything about contracts, employment law, tenant rights, consumer protection, IP, and more.\n\nTip: type /correct followed by a better answer to override my response for similar questions.",
    sources: [],
    correctionApplied: false,
  }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const chatRef = useRef(null);
  const inputRef = useRef(null);

  const isCorrectCommand = input.toLowerCase().startsWith("/correct");

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, loading]);

  const getHistory = () => messages
    .slice(1)
    .filter(m => !m.content.startsWith("/correct"))
    .map(m => ({ role: m.role === "assistant" ? "assistant" : "user", content: m.content }));

  async function sendMessage(text) {
    const query = (text || input).trim();
    if (!query || loading) return;

    setInput("");
    setShowSuggestions(false);
    setMessages(prev => [...prev, { role: "user", content: query, sources: [] }]);
    setLoading(true);

    try {
      const { data } = await axios.post(`${API_URL}/chat`, {
        query,
        session_id: SESSION_ID,
        conversation_history: getHistory(),
      });

      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        correctionApplied: data.correction_applied || false,
      }]);

    } catch {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Sorry, couldn't connect to the backend. Make sure the FastAPI server is running on port 8000.",
        sources: [],
        correctionApplied: false,
      }]);
    }

    setLoading(false);
    inputRef.current?.focus();
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }

  function handleInputChange(e) {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#f5f5f0", fontFamily: "system-ui, sans-serif" }}>
      <style>{`
        @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }
        textarea:focus { outline: none; border-color: #6366f1 !important; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{ background: "#1e1b4b", color: "white", padding: "14px 20px", display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{ width: "36px", height: "36px", background: "#4f46e5", borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px" }}>⚖</div>
        <div>
          <div style={{ fontSize: "17px", fontWeight: "600" }}>LegalBot</div>
          <div style={{ fontSize: "12px", color: "#a5b4fc" }}>Wikipedia-powered Legal FAQ Assistant</div>
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{ background: "#fef3c7", borderBottom: "1px solid #fde68a", padding: "7px 20px", fontSize: "12px", color: "#92400e", textAlign: "center" }}>
        For informational purposes only — not legal advice. Consult a qualified attorney for your situation.
      </div>

      {/* Chat */}
      <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: "20px", display: "flex", flexDirection: "column", gap: "16px" }}>
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && <TypingIndicator />}
      </div>

      {/* Suggestions */}
      {showSuggestions && (
        <div style={{ padding: "0 20px 10px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {SUGGESTIONS.map((s, i) => (
            <button key={i} onClick={() => sendMessage(s)} style={{
              background: "white", border: "1px solid #d1d5db", borderRadius: "20px",
              padding: "7px 14px", fontSize: "13px", cursor: "pointer", color: "#374151",
            }}>{s}</button>
          ))}
        </div>
      )}

      {/* /correct hint shown when user starts typing the command */}
      <CommandHint visible={isCorrectCommand} />

      {/* Input */}
      <div style={{
        padding: "12px 20px 16px", background: "white",
        borderTop: isCorrectCommand ? "1px solid #fde68a" : "1px solid #e5e7eb",
        display: "flex", gap: "10px", alignItems: "flex-end",
        transition: "border-color 0.2s",
      }}>
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder='Ask a legal question, or type /correct to override an answer...'
          style={{
            flex: 1, borderRadius: "12px", padding: "10px 14px",
            fontSize: "14px", fontFamily: "inherit", resize: "none",
            maxHeight: "120px", lineHeight: "1.5",
            border: isCorrectCommand ? "1px solid #f59e0b" : "1px solid #d1d5db",
            background: isCorrectCommand ? "#fefce8" : "white",
            transition: "all 0.2s",
          }}
        />
        <button
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          style={{
            background: loading || !input.trim()
              ? "#c7d2fe"
              : isCorrectCommand ? "#f59e0b" : "#4f46e5",
            color: "white", border: "none", borderRadius: "10px",
            width: "40px", height: "40px",
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0, transition: "background 0.15s",
          }}
        >
          {isCorrectCommand
            ? <span style={{ fontSize: "16px" }}>✎</span>
            : <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
          }
        </button>
      </div>
    </div>
  );
}