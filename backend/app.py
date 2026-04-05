import os
import uuid
import math
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import chromadb

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K_CHUNKS = 4

# Set to None for no expiry, or a number for hours e.g. 24
CORRECTION_TTL_HOURS = None

FOLLOWUP_SIMILARITY_THRESHOLD = 0.5
CORRECTION_SIMILARITY_THRESHOLD = 0.9
CORRECTION_TOP_K = 5

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Legal KB Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session state ──────────────────────────────────────────────────────────────

last_retrieved_chunks: dict[str, dict] = {}

# ── Load embedding model ───────────────────────────────────────────────────────

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Load ChromaDB — two collections ───────────────────────────────────────────

print("Loading ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

vectorstore = Chroma(
    client=chroma_client,
    embedding_function=embeddings,
    collection_name="legal_kb",
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K_CHUNKS},
)
print(f"  legal_kb loaded — {vectorstore._collection.count()} vectors")

corrections_store = Chroma(
    client=chroma_client,
    embedding_function=embeddings,
    collection_name="corrections",
)
print(f"  corrections loaded — {corrections_store._collection.count()} corrections")

# ── Load Groq LLM ──────────────────────────────────────────────────────────────

print("Connecting to Groq...")
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

# ── Prompts ────────────────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are LegalBot, a knowledgeable legal FAQ assistant.
Answer the question using ONLY the context provided below.
You may also use the chat history to understand follow-up questions.
Only answer legal questions. If the question is not related to law,
politely say you can only help with legal topics.
If the context does not contain enough information, say:
"I don't have enough information on that topic. Please consult a qualified attorney."
Always remind the user this is for informational purposes only and not legal advice.

Chat history:
{chat_history}

Context:
{context}

Question: {question}

Answer:""",
)

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
Given the conversation history and a follow-up question,
rephrase the follow-up into a standalone question using the history for context.
If it's already standalone, return it as-is.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:""",
)

# ── ConversationalRetrievalChain ───────────────────────────────────────────────

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    condense_question_prompt=CONDENSE_PROMPT,
    verbose=False,
)

# ── Request / Response models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    conversation_history: Optional[list[Message]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    correction_applied: bool = False

# ── Helpers ────────────────────────────────────────────────────────────────────

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def is_followup(new_query: str, last_query: str) -> bool:
    vec1 = embeddings.embed_query(new_query)
    vec2 = embeddings.embed_query(last_query)
    similarity = cosine_similarity(vec1, vec2)
    print(f"  Followup similarity: {similarity:.3f} (threshold: {FOLLOWUP_SIMILARITY_THRESHOLD})")
    return similarity >= FOLLOWUP_SIMILARITY_THRESHOLD


def build_chat_history(history: list[Message]):
    lc_history = []
    for msg in history:
        if msg.role == "user":
            lc_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            lc_history.append(AIMessage(content=msg.content))
    return lc_history


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_expires_at() -> str:
    """
    Returns expiry timestamp string based on CORRECTION_TTL_HOURS.
    Returns "never" if TTL is None.
    """
    if CORRECTION_TTL_HOURS is None:
        return "never"
    return (now_utc() + timedelta(hours=CORRECTION_TTL_HOURS)).isoformat()


def is_expired(expires_at_str: str) -> bool:
    """
    Returns True if the correction has expired.
    Returns False if expires_at is "never" or still in the future.
    """
    if expires_at_str == "never":
        return False
    expires_at = datetime.fromisoformat(expires_at_str)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < now_utc()


def purge_expired_corrections():
    """
    Remove corrections that have passed their TTL.
    Skips corrections with expires_at = "never".
    """
    try:
        all_corrections = corrections_store.get()
        if not all_corrections["ids"]:
            return
        expired_ids = [
            all_corrections["ids"][i]
            for i, meta in enumerate(all_corrections["metadatas"])
            if is_expired(meta["expires_at"])
        ]
        if expired_ids:
            corrections_store.delete(ids=expired_ids)
            print(f"  Purged {len(expired_ids)} expired correction(s)")
    except Exception as e:
        print(f"  Purge error: {e}")


def get_chunk_ids(source_docs: list) -> list[str]:
    """
    Extract stable chunk IDs from retrieved documents.
    Reads from doc.metadata["chunk_id"] set at ingestion time.
    """
    return [
        doc.metadata.get("chunk_id", doc.page_content[:40])
        for doc in source_docs
    ]


def find_best_correction(chunk_ids: list[str], current_query: str) -> dict | None:
    """
    Search corrections collection for a matching correction.
    Only considers non-expired corrections above similarity threshold
    whose chunk IDs are a subset of the currently retrieved chunks.
    """
    purge_expired_corrections()

    if corrections_store._collection.count() == 0:
        return None

    try:
        results = corrections_store.similarity_search_with_relevance_scores(
            current_query,
            k=min(CORRECTION_TOP_K, corrections_store._collection.count()),
        )
    except Exception as e:
        print(f"  Correction search error: {e}")
        return None

    if not results:
        return None

    chunk_id_set = set(chunk_ids)
    matches = []

    for doc, score in results:
        meta = doc.metadata

        # Skip expired
        if is_expired(meta["expires_at"]):
            continue

        # Chunk ID overlap check
        correction_chunk_ids = set(meta["chunk_ids"].split(","))
        if not correction_chunk_ids.issubset(chunk_id_set):
            continue

        # Query similarity threshold
        print(f"  Correction candidate similarity: {score:.3f} (threshold: {CORRECTION_SIMILARITY_THRESHOLD})")
        if score < CORRECTION_SIMILARITY_THRESHOLD:
            continue

        matches.append({
            "corrected_answer": meta["corrected_answer"],
            "original_query": meta["original_query"],
            "chunk_ids": correction_chunk_ids,
            "similarity": score,
        })

    if not matches:
        return None

    return min(matches, key=lambda c: len(c["chunk_ids"]))

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Legal KB Chatbot API is running"}


@app.get("/health")
def health():
    purge_expired_corrections()
    return {
        "status": "ok",
        "vectors_in_db": vectorstore._collection.count(),
        "corrections_in_db": corrections_store._collection.count(),
        "model": GROQ_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or "default"

    # ── Handle /correct command ────────────────────────────────────────────────
    if query.lower().startswith("/correct "):
        corrected_answer = query[len("/correct "):].strip()

        if not corrected_answer:
            return ChatResponse(
                answer="Please provide the corrected answer after `/correct`.\n\nExample:\n`/correct A contract always requires consideration to be valid.`",
                sources=[],
                correction_applied=False,
            )

        session_data = last_retrieved_chunks.get(session_id)
        if not session_data:
            return ChatResponse(
                answer="No recent query found to correct. Please ask a question first, then use `/correct` to override the answer.",
                sources=[],
                correction_applied=False,
            )

        chunk_ids = session_data["chunk_ids"]
        original_query = session_data["query"]
        correction_id = str(uuid.uuid4())[:8]

        corrections_store.add_texts(
            texts=[original_query],
            metadatas=[{
                "original_query": original_query,
                "corrected_answer": corrected_answer,
                "chunk_ids": ",".join(chunk_ids),
                "expires_at": get_expires_at(),
            }],
            ids=[correction_id],
        )

        print(f"  Correction saved: id={correction_id}, query='{original_query}', expires={get_expires_at()}")

        ttl_msg = "This answer will never expire unless manually deleted or approved." if CORRECTION_TTL_HOURS is None else f"This answer will apply for the next {CORRECTION_TTL_HOURS} hours."

        return ChatResponse(
            answer=f"✓ Correction submitted for review.\n\nOriginal query: `{original_query}`\nCorrection: \"{corrected_answer}\"\n\n{ttl_msg}",
            sources=[],
            correction_applied=False,
        )

    # ── Normal query flow ──────────────────────────────────────────────────────

    chat_history = build_chat_history(request.conversation_history or [])

    session_data = last_retrieved_chunks.get(session_id)
    last_query = session_data["query"] if session_data else ""

    if last_query and is_followup(query, last_query):
        print(f"  Follow-up to: '{last_query}'")
        effective_history = chat_history
    else:
        print(f"  Fresh topic — skipping condense")
        effective_history = []

    result = qa_chain.invoke({
        "question": query,
        "chat_history": effective_history,
    })

    source_docs = result.get("source_documents", [])
    chunk_ids = get_chunk_ids(source_docs)

    last_retrieved_chunks[session_id] = {
        "chunk_ids": chunk_ids,
        "query": query,
    }

    correction = find_best_correction(chunk_ids, query)
    if correction:
        print(f"  Correction applied from: '{correction['original_query']}'")
        return ChatResponse(
            answer=correction["corrected_answer"],
            sources=list({
                doc.metadata.get("title", "Unknown")
                for doc in source_docs
            }),
            correction_applied=True,
        )

    sources = list({
        doc.metadata.get("title", doc.metadata.get("topic", "Unknown"))
        for doc in source_docs
    })

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        correction_applied=False,
    )


@app.get("/corrections")
def list_corrections():
    """List all active non-expired corrections."""
    purge_expired_corrections()
    try:
        all_corrections = corrections_store.get()
        if not all_corrections["ids"]:
            return {"count": 0, "corrections": []}

        active = []
        for i, meta in enumerate(all_corrections["metadatas"]):
            if not is_expired(meta["expires_at"]):
                active.append({
                    "id": all_corrections["ids"][i],
                    "original_query": meta["original_query"],
                    "corrected_answer": meta["corrected_answer"],
                    "chunk_count": len(meta["chunk_ids"].split(",")),
                    "expires_at": meta["expires_at"],
                })
        return {"count": len(active), "corrections": active}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/corrections/{correction_id}/approve")
def approve_correction(correction_id: str):
    """
    Approve a correction — moves it from the corrections collection
    into legal_kb as a permanent chunk, then deletes it from corrections.

    After approval:
    - The corrected answer becomes a permanent chunk in legal_kb
    - It is retrieved naturally by similarity search like any Wikipedia chunk
    - The correction lookup layer is no longer needed for this answer
    - The original correction is removed from the corrections collection
    """
    try:
        # Step 1 — fetch the correction from corrections collection
        result = corrections_store.get(ids=[correction_id])

        if not result["ids"]:
            raise HTTPException(status_code=404, detail="Correction not found")

        meta = result["metadatas"][0]
        corrected_answer = meta["corrected_answer"]
        original_query = meta["original_query"]

        # Step 2 — insert corrected answer into legal_kb as a permanent chunk
        new_chunk_id = str(uuid.uuid4())

        vectorstore.add_texts(
            texts=[corrected_answer],
            metadatas=[{
                "chunk_id":    new_chunk_id,
                "title":       "Approved Correction",
                "topic":       original_query,
                "source":      "correction",
                "approved_at": now_utc().isoformat(),
                "original_query": original_query,
            }],
            ids=[new_chunk_id],
        )

        print(f"  Correction approved: moved to legal_kb as chunk {new_chunk_id}")
        print(f"  legal_kb now has {vectorstore._collection.count()} vectors")

        # Step 3 — delete from corrections collection
        corrections_store.delete(ids=[correction_id])

        print(f"  Correction {correction_id} removed from corrections collection")

        return {
            "message": f"Correction approved and permanently added to knowledge base.",
            "original_query": original_query,
            "corrected_answer": corrected_answer,
            "new_chunk_id": new_chunk_id,
            "legal_kb_total": vectorstore._collection.count(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/correct/{correction_id}")
def delete_correction(correction_id: str):
    """Reject and delete a correction by ID."""
    try:
        corrections_store.delete(ids=[correction_id])
        return {"message": f"Correction {correction_id} rejected and deleted"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Correction not found: {e}")