# LegalBot ⚖️

A full-stack AI chatbot that answers legal questions using a **Retrieval-Augmented Generation (RAG)** pipeline. Legal knowledge is sourced from Wikipedia, chunked, embedded, and stored in ChromaDB. Queries are answered by LLaMA 3.3 70B via Groq, grounded in retrieved context. Includes a human-in-the-loop correction layer that lets domain experts override answers, with a reviewer approval workflow that permanently promotes corrections into the knowledge base.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API (free) |
| RAG Framework | LangChain |
| Vector Database | ChromaDB (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (free, local) |
| Knowledge Source | Wikipedia (via LangChain WikipediaLoader) |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + Vite |

---

## Project Structure

```
legal-kb-bot/
├── backend/
│   ├── app.py              # FastAPI server — RAG pipeline + correction layer
│   ├── ingest.py           # One-time ingestion: Wikipedia → ChromaDB
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variable template
└── frontend/
    └── src/
        ├── App.jsx         # React chat UI
        └── main.jsx        # React entry point
```

---

## Prerequisites

Make sure you have the following installed before starting:

- **Python 3.10+** — [Download](https://www.python.org/downloads/)
- **Node.js 18+** — [Download](https://nodejs.org/)
- **Git** — [Download](https://git-scm.com/)

---

## Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Toshin2002/legal-kb-bot.git
cd legal-kb-bot
```

### Step 2 — Set up the Python backend

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Get your free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account (no credit card needed)
3. Click **API Keys** → **Create API Key**
4. Copy the key

### Step 4 — Configure environment variables

Create a `.env` file inside the `backend/` folder:

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5 — Build the knowledge base

This fetches legal articles from Wikipedia, chunks them, embeds them, and stores them in ChromaDB. Run this once — it takes 2–5 minutes depending on your internet speed.

```bash
# Make sure you're in the backend/ folder with venv activated
python ingest.py
```

You should see output like:

```
Fetching 10 topics from Wikipedia...
  Loading: Contract law
    -> 1 article(s)
  Loading: Tort law
    -> 1 article(s)
  ...
Total chunks created: ~120
Storing vectors in ChromaDB...
Done! 120 vectors stored.
Ingestion complete!
```

A `chroma_db/` folder will appear in `backend/` — this is your local vector database.

### Step 6 — Start the backend server

```bash
# Still in backend/ with venv activated
uvicorn app:app --port 8000
```

You should see:

```
Loading embedding model...
Loading ChromaDB...
  legal_kb loaded — 120 vectors
  corrections loaded — 0 corrections
Connecting to Groq...
INFO: Uvicorn running on http://127.0.0.1:8000
```

### Step 7 — Set up and start the frontend

Open a **new terminal** (keep the backend running):

```bash
cd frontend
npm install
npm run dev
```

You should see:

```
  VITE v5.x  ready in 300ms
  ➜  Local:   http://localhost:5173/
```

### Step 8 — Open the app

Open **http://localhost:5173** in your browser. You should see the LegalBot chat interface.

---

## How to Use

### Asking questions

Type any legal question and press **Enter**:

```
What is a contract?
Can my landlord evict me without notice?
What are my rights as an employee?
What is small claims court?
```

### Follow-up questions

The bot remembers context within a conversation:

```
User: What is a contract?
Bot:  A contract is a legally binding agreement...

User: Does it need to be in writing?   ← the bot understands "it" refers to a contract
Bot:  Not always. Verbal contracts are enforceable...
```

### Correcting an answer

If the bot gives an answer you want to override, type `/correct ` followed by your preferred answer:

```
/correct A contract is an agreement that requires offer, acceptance, and consideration.
```

The correction will apply to any similar question going forward.

### Reviewing corrections

Visit **http://localhost:8000/docs** to access the interactive API docs. From there you can:

- `GET /corrections` — see all pending corrections
- `POST /corrections/{id}/approve` — approve a correction (moves it permanently into the knowledge base)
- `DELETE /correct/{id}` — reject and delete a correction

---

## Adding More Topics

To expand the knowledge base, edit `LEGAL_TOPICS` in `backend/ingest.py`:

```python
LEGAL_TOPICS = [
    "Contract law",
    "Tort law",
    "Your new topic here",   # ← add topics
    ...
]
```

Then delete the old database and re-run ingestion:

```bash
# Windows
Remove-Item -Recurse -Force chroma_db

# Mac/Linux
rm -rf chroma_db

python ingest.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Vector counts + model info |
| POST | `/chat` | Main chat endpoint |
| GET | `/corrections` | List all active corrections |
| POST | `/corrections/{id}/approve` | Approve and permanently add correction to knowledge base |
| DELETE | `/correct/{id}` | Reject and delete a correction |

Full interactive docs available at **http://localhost:8000/docs** while the server is running.

---

## Troubleshooting

**`ModuleNotFoundError`**
Make sure your virtual environment is activated (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on Mac/Linux) before running any Python commands.

**`groq.BadRequestError: model decommissioned`**
Groq occasionally deprecates models. Check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the latest available models and update `GROQ_MODEL` in `backend/app.py`.

**`Failed to send telemetry event`**
This is a harmless ChromaDB internal warning — it does not affect functionality and can be safely ignored.

**`uvicorn: command not found`**
Your virtual environment is not activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) first.

**Frontend can't connect to backend**
Make sure the backend is running on port 8000 before starting the frontend. Check `API_URL` in `frontend/src/App.jsx` matches your backend address.

**Corrections not persisting after server restart**
Make sure you re-ran `ingest.py` after the stable chunk ID fix. If you have an old `chroma_db/` folder, delete it and run `python ingest.py` again.

---

## License

MIT License — feel free to use, modify, and distribute.
