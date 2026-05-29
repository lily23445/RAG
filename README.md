# RAG
# Chat with PDF — RAG from Scratch

🔗 Live Demo(so far--day 3): [https://askmeque.streamlit.app/]
A document Q&A app built with Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions in natural language, and get answers grounded in the document — with source page references.

Built as a focused 7-day learning sprint to go from zero to a live, deployed project.

---

## What it does

1. You give it a PDF.
2. It reads the PDF, breaks it into chunks, and converts each chunk into a vector that captures its meaning.
3. When you ask a question, it finds the most relevant chunks and sends them to an LLM.
4. The LLM answers using **only** that retrieved context — and tells you which pages the answer came from.

This is the same pattern behind most "chat with your documents" products.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Framework | LangChain (v1, LCEL) | Standard interfaces, easy to swap components |
| LLM | Groq — Llama 3.3 70B | Free tier, very fast inference |
| Embeddings | sentence-transformers MiniLM (local) | Free, runs on CPU, no API needed |
| Vector store | FAISS | Fast in-memory similarity search, zero setup |
| PDF loading | PyPDFLoader | Simple page-by-page text extraction |

No paid API keys required — the whole pipeline runs on free tiers.

---

## The 7-day plan

- **Day 1 — Foundation** ✅ Core RAG pipeline working in the terminal. Load → split → embed → store → retrieve → answer.
- **Day 2 — Make it real** 🔨 Persist the vector index, return source citations, harden the prompt against hallucination, interactive loop.
- **Day 3 — Polish** Conversation memory for follow-up questions; UI** Wrap it in a Streamlit web interface with file upload and a chat box.
- **Day 4 — handle multiple files.
- **Day 5 — Quality** Tune retrieval, improve prompts, add evaluation.
- **Day 6 — Deploy** Push it live with a public URL.
- **Day 7 — Ship** Polish the README, record a demo, write it up.

---

## Day 1 — what's working

The core engine runs end to end. On a 42-page test PDF it:

- Loaded all 42 pages
- Split them into 111 overlapping chunks (1000 chars each, 200-char overlap)
- Embedded every chunk into a 384-dimension vector using MiniLM
- Indexed the vectors in FAISS
- Answered questions accurately, pulling from the right chunks

The retrieval chain is built with LangChain Expression Language (LCEL):

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Setup

```bash
# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# install dependencies
pip install langchain langchain-core langchain-community langchain-text-splitters \
            langchain-groq langchain-huggingface sentence-transformers \
            faiss-cpu pypdf python-dotenv
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

---

## Run

```bash
python main.py
```

# Chat with PDF — RAG from Scratch

A document Q&A app built with Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions in natural language, and get answers grounded in the document — with source page references.

Built as a focused 7-day learning sprint to go from zero to a live, deployed project.

---

## What it does

1. You give it a PDF.
2. It reads the PDF, breaks it into chunks, and converts each chunk into a vector that captures its meaning.
3. When you ask a question, it finds the most relevant chunks and sends them to an LLM.
4. The LLM answers using **only** that retrieved context — and tells you which pages the answer came from.

This is the same pattern behind most "chat with your documents" products.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Framework | LangChain (v1, LCEL) | Standard interfaces, easy to swap components |
| LLM | Groq — Llama 3.3 70B | Free tier, very fast inference |
| Embeddings | sentence-transformers MiniLM (local) | Free, runs on CPU, no API needed |
| Vector store | FAISS | Fast in-memory similarity search, zero setup |
| PDF loading | PyPDFLoader | Simple page-by-page text extraction |

No paid API keys required — the whole pipeline runs on free tiers.

---

## The 7-day plan

- **Day 1 — Foundation** ✅ Core RAG pipeline working in the terminal. Load → split → embed → store → retrieve → answer.
- **Day 2 — Make it real** ✅ Persistent vector index, source citations, hardened prompt against hallucination, interactive loop.
- **Day 3 — Polish** 🔨 Conversation memory for follow-up questions; smarter handling of multiple documents.
- **Day 4 — UI** Wrap it in a Streamlit web interface with file upload and a chat box.
- **Day 5 — Quality** Tune retrieval, improve prompts, add evaluation.
- **Day 6 — Deploy** Push it live with a public URL.
- **Day 7 — Ship** Polish the README, record a demo, write it up.

---

## Day 1 — what's working

The core engine runs end to end. On a 42-page test PDF it:

- Loaded all 42 pages
- Split them into 111 overlapping chunks (1000 chars each, 200-char overlap)
- Embedded every chunk into a 384-dimension vector using MiniLM
- Indexed the vectors in FAISS
- Answered questions accurately, pulling from the right chunks

The retrieval chain is built with LangChain Expression Language (LCEL):

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## Day 2 — what's working

Three upgrades that make it usable and demo-ready:

- **Persistence** — the FAISS index is saved to disk (`save_local` / `load_local`). The first run builds and embeds; every run after loads the index in a fraction of a second instead of re-embedding the whole PDF.
- **Source citations** — every answer lists the PDF pages it drew from, with a short preview snippet. Uses `RunnableParallel` to fetch the answer and the source documents in one pass. Duplicate pages are de-duplicated.
- **Hardened prompt** — a strict, rule-based system prompt that answers only from the retrieved context and refuses with a fixed phrase when the document doesn't cover the question. This is the main defense against hallucination.
- **Interactive loop** — ask questions continuously in the terminal until you type `quit`.

A note on a real issue solved along the way: the retriever always returns its top-k nearest chunks regardless of true relevance, so even irrelevant questions surfaced sources. This is handled with a refusal check that suppresses sources when the model declines to answer. A more principled approach (a similarity score threshold) was explored but deferred — it surfaced that FAISS defaults to L2 distance, which produces out-of-range relevance scores with normalized embeddings unless switched to cosine similarity.

---

## Day 3 — Polish ✅

The project evolved from a terminal prototype into a full conversational document assistant.

What was added

Conversation memory — follow-up questions now work naturally. Instead of treating every question independently, the app keeps track of chat history so users can ask contextual questions like:

“Summarize chapter 2.”
“Now explain that in simpler words.”
“What was the author’s conclusion again?”

This was implemented using LangChain memory integrated into the LCEL pipeline.


Streamlit UI — the terminal app was converted into a web interface with:

PDF upload support
Chat-style conversation window
Persistent chat history
Source citation display
Cleaner user experience overall
## Setup

```bash
# create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# install dependencies
pip install langchain langchain-core langchain-community langchain-text-splitters \
            langchain-groq langchain-huggingface sentence-transformers \
            faiss-cpu pypdf python-dotenv
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

---

## Run

```bash
python main.py
```

Set `PDF_PATH` in `main.py` to point at your PDF first. The first run builds the index; later runs load it from the `faiss_index/` folder. (Switching to a new PDF? Delete the `faiss_index/` folder so it rebuilds.)

---

## Concepts learned

**Day 1**
- **RAG** — why we retrieve context at query time instead of fine-tuning
- **Embeddings** — converting text to vectors that encode meaning
- **Chunking & overlap** — why we split documents and why chunks overlap
- **Vector search** — how FAISS finds nearest neighbors using distance metrics
- **LCEL** — composing pipeline steps with the pipe operator, where every step is a Runnable

**Day 2**
- **Persistence** — saving and loading a vector index so embeddings aren't recomputed every run
- **RunnableParallel** — running two branches on the same input to return answer + sources together
- **Prompt hardening** — strict grounding rules to prevent hallucination
- **Retrieval has no "I don't know"** — the retriever always returns top-k; relevance judgment lives in the LLM or a score threshold, not the retriever
- **Distance metrics** — L2 vs cosine, and why the choice affects relevance scores

---
## Day 3 — Polish ✅

- ** Conversational RAG
- ** Chat history & memory management
- ** Stateful vs stateless interactions
- ** Streamlit app architecture
- ** Handling multiple document sources
- ** Deploying  applications publicly



## Status

🚧 Work in progress — building one day at a time. Follow the commit history to watch it come together.
