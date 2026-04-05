from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Configuration ──────────────────────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"      # Where ChromaDB will be saved locally
CHUNK_SIZE = 500                # Characters per chunk
CHUNK_OVERLAP = 80              # Overlap between chunks (keeps context continuous)
DOCS_PER_TOPIC = 1              # Wikipedia articles to fetch per topic

# Legal topics to load from Wikipedia
LEGAL_TOPICS = [
    "Contract law",
    "Tort law",
    "Landlord-tenant law",
    "Consumer protection",
    "Intellectual property",
    "Privacy law",
    "Small claims court",
    "At-will employment",
    "Fair Labor Standards Act",
    "Fair Debt Collection Practices Act",
]

# Free local embedding model — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Step 1: Load articles from Wikipedia ──────────────────────────────────────

def load_wikipedia_docs(topics: list) -> list:
    all_docs = []
    print(f"\nFetching {len(topics)} topics from Wikipedia...\n")

    for topic in topics:
        print(f"  Loading: {topic}")
        try:
            loader = WikipediaLoader(
                query=topic,
                load_max_docs=DOCS_PER_TOPIC,
                doc_content_chars_max=6000,
            )
            docs = loader.load()

            for doc in docs:
                doc.metadata["topic"] = topic

            all_docs.extend(docs)
            print(f"    -> {len(docs)} article(s), ~{sum(len(d.page_content) for d in docs)} chars")

        except Exception as e:
            print(f"    x Failed: {e}")

    print(f"\nTotal articles loaded: {len(all_docs)}")
    return all_docs


# ── Step 2: Split into chunks ──────────────────────────────────────────────────

def chunk_documents(docs: list) -> list:
    print("\nSplitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks


# ── Step 3: Embed and store in ChromaDB ───────────────────────────────────────

def build_vector_store(chunks: list) -> Chroma:
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(This may take a minute the first time...)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Storing vectors in ChromaDB at: {CHROMA_DIR}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="legal_kb",
    )

    vectorstore.persist()
    count = vectorstore._collection.count()
    print(f"\nDone! {count} vectors stored in ChromaDB.")
    return vectorstore


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Legal KB — Ingestion Pipeline")
    print("=" * 50)

    docs = load_wikipedia_docs(LEGAL_TOPICS)
    chunks = chunk_documents(docs)
    build_vector_store(chunks)

    print("\nIngestion complete!")
    print("Next: run `python app.py` to start the chatbot.")
    print("=" * 50)


if __name__ == "__main__":
    main()