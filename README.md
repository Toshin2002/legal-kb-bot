# legal-kb-bot
A full-stack AI chatbot that answers legal questions using a 
Retrieval-Augmented Generation (RAG) pipeline. Legal knowledge 
is sourced from Wikipedia, chunked, embedded, and stored in 
ChromaDB. Queries are answered by LLaMA 3.3 70B via Groq, 
grounded in retrieved context. Includes a correction layer 
that lets domain experts override answers, with a reviewer 
approval workflow that permanently promotes corrections into 
the knowledge base.
