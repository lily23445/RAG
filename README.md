# Chat with PDF — RAG from Scratch

🔗 Live Demo: [https://askmeque.streamlit.app/]

A document Q&A app built with Retrieval-Augmented Generation (RAG). Upload one or more PDFs, ask questions in natural language, and get answers grounded in the document — with source page references and full retrieval transparency.

Built as a focused 7-day learning sprint to go from zero to a production-aware, live deployed project.

---

## What it does

1. You upload one or more PDFs.
2. It reads them, breaks them into chunks, and converts each chunk into a vector that captures its meaning.
3. When you ask a question, it finds the 8 most relevant chunks using FAISS, then reranks them with Cohere's cross-encoder to pick the best 4.
4. The LLM answers using **only** that retrieved context — and tells you which pages the answer came from.
5. Follow-up questions work naturally — vague references like "what are its disadvantages?" are automatically resolved using conversation history before retrieval.
6. The full pipeline is visible — see what was searched, what FAISS returned, and how Cohere reranked it.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Framework | LangChain (v1, LCEL) | Standard interfaces, easy to swap components |
| LLM | Groq — Llama 3.3 70B | Free tier, very fast inference |
| Embeddings | sentence-transformers MiniLM (local) | Free, runs on CPU, no API needed |
| Vector store | FAISS | Fast in-memory similarity search, zero setup |
| Reranker | Cohere Rerank v3 | Cross-encoder reranking for more accurate retrieval |
| PDF loading | PyPDFLoader + pdfplumber | Text extraction + structured table extraction |
| UI | Streamlit | Fast, clean web interface |

---

## The 7-day plan

- **Day 1 — Foundation** ✅ Core RAG pipeline in the terminal. Load → split → embed → store → retrieve → answer.
- **Day 2 — Make it real** ✅ Persistent vector index, source citations, hardened prompt against hallucination, interactive loop.
- **Day 3 — Polish** ✅ Conversation memory, query reformulation, Streamlit UI, multi-PDF support, deployed live.
- **Day 4 — Hardening** ✅ Failure handling, Groq rate limit graceful degradation, query analysis agent replaced with smarter answer prompt clarification, PDF caching.
- **Day 5 — Evaluation & Reranking** ✅ Evaluated retrieval quality, identified top-k gaps, added Cohere reranking, built retrieval debug UI to visualize the full pipeline.
- **Day 6 — LangSmith evaluation, table extraction & code hardening** ✅ LLM-as-judge faithfulness scoring, pdfplumber table extraction, constants, empty-retrieval short-circuit, silent exception fixes.
- **Day 7 — Ship** Final cleanup and submission.

---

## Day 1 — Foundation

The core engine runs end to end. On a 42-page test PDF it:

- Loaded all 42 pages
- Split them into 111 overlapping chunks (1000 chars each, 200-char overlap)
- Embedded every chunk into a 384-dimension vector using MiniLM
- Indexed the vectors in FAISS
- Answered questions accurately, pulling from the right chunks

```python

rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)
```
---

## Day 2 — Make it real

- **Persistence** — FAISS index saved to disk. First run builds and embeds; every run after loads instantly.
- **Source citations** — every answer lists the PDF pages it drew from with a short preview snippet.
- **Hardened prompt** — strict system prompt that answers only from retrieved context and refuses with a fixed phrase when the document doesn't cover the question.
- **Interactive loop** — continuous terminal Q&A until you type `quit`.

A real issue solved: the retriever always returns top-k chunks regardless of relevance — so even irrelevant questions surfaced sources. Fixed with a refusal check that suppresses sources when the model declines to answer. Also explored FAISS distance metrics — L2 vs cosine, and why the choice affects relevance scores with normalized embeddings.

---

## Day 3 — Polish

- **Conversational memory** — follow-up questions work naturally using LangChain chat history.
- **Query reformulation** — vague follow-ups like *"what are its disadvantages?"* are rewritten into self-contained queries like *"disadvantages of the Suzuki-Kasami algorithm"* before hitting the retriever.
- **Multi-PDF support** — upload and query across multiple documents simultaneously.
- **Streamlit UI** — PDF upload, chat window, source citations, clear conversation button.
- **Deployed live** on Streamlit Cloud.

---

## Day 4 — Hardening

### Failure handling
Production systems fail. Two failure modes addressed:

- **Groq rate limit** — when the daily token limit is hit, the app returns a clean user-facing message instead of crashing with a stack trace.
- **JSON parsing failures** — the query analysis agent occasionally returned malformed JSON. Added a safe fallback that defaults to `answerable` so the pipeline never blocks on an agent error.

### Query analysis agent — replaced
Earlier versions used a dedicated Query Analysis Agent: a separate LLM call on every question just to decide *"is this clear enough to retrieve?"*. This was wasteful — especially on follow-up questions where the answer was always yes.

Replaced by moving clarification logic directly into the answer prompt. The LLM now sees both the question and the retrieved context together, and decides whether to answer or ask for clarification in one step. Result: one fewer LLM call on every single query.

Before: Analysis Agent → Reformulation → Retrieval → Answer  (3 LLM calls)
After:  Reformulation → Retrieval → Answer                    (2 LLM calls)

### PDF caching
PDFs are hashed on upload (MD5). If the same PDF is uploaded again, the pre-built FAISS index loads from disk instantly — no re-embedding.

---

## Day 5 — Evaluation & Reranking

### The problem — top-k isn't always enough
Evaluation revealed that FAISS retrieval was working — but the most important chunk wasn't always making it into the top results. Simply increasing `k` helped but wasn't reliable. A higher `k` also means more noise in the context sent to the LLM.

### The fix — reranking
Fetch wide with FAISS (`k=8`), rerank precisely with Cohere's cross-encoder (`top_n=4`).

The key difference:
- **FAISS** compares embedding vectors — fast but approximate. It measures how similar two vectors are in space, not how relevant a chunk is to a specific question.
- **Cohere cross-encoder** reads the query and chunk *together* and scores their relevance directly — much more accurate, but too slow to run on the entire index. So we use FAISS to narrow the field and Cohere to pick the best from that shortlist.

FAISS (k=8) → Cohere Rerank (top_n=4) → LLM

### Retrieval debug UI
To verify reranking was actually improving results, a debug panel was added to the UI. Every response now shows:

- **Reformulated query** — what was actually sent to the retriever (only shown when the question was rewritten from history)
- **FAISS top-8** — raw retrieval results before reranking, with page numbers and snippet previews
- **Cohere top-4** — final results after reranking, with rank change indicators showing where reranking moved chunks up or down

❓ You asked:       "what are its disadvantages?"
🔄 Searched for:    "disadvantages of the Suzuki-Kasami algorithm"
📄 FAISS returned:  Page 3, 7, 9, 12, 14, 18, 21, 25
🎯 Cohere kept:     Page 7 (was #3 → now #1 🔺), Page 3, Page 14 (was #5 → now #3 🔺), Page 9
💬 Answer:          ...

This makes the entire RAG pipeline visible and verifiable — not a black box.

---

## Day 6 — LangSmith evaluation, table extraction & code hardening

### Evaluation with LangSmith
With the pipeline working end-to-end, Day 6 focused on **measuring** it — not eyeballing answers, but systematically evaluating whether the RAG system produced faithful, grounded responses.

**Test document:** Distributed Computing exam notes (TE CSE-AIML, Sem VI, APSIT)

Questions were run through the full pipeline — reformulation → FAISS retrieval → Cohere reranking → LLM answer. LangSmith captured every run as a **trace**, making the full input/output at each step visible in the dashboard.

An **LLM-as-judge evaluator** then scored each answer for faithfulness: it reads the retrieved context and generated answer together and decides whether the answer is supported — no ground-truth answers needed.

> Pipeline: Question → Reformulation → FAISS (k=8) → Cohere Rerank (top 4) → LLM Answer → LangSmith Trace → Judge Score (1–5)

### Evaluator prompt

```
Persona: Expert QA evaluator assessing RAG answer quality.

Rubric — a high-quality answer:
  - Is grounded in the PDF content (every claim traceable to source)
  - Directly and completely answers the question without padding
  - Does not hallucinate facts absent from retrieved context
  - Cites or references the relevant section/page when applicable

Deduct points for:
  - Hallucination (info not in retrieved context)
  - Ignoring part of the question
  - Filler phrases ("Great question!", "Hope this helps")
  - Vague answers that could apply to any document

Output: Score 1–5 + one-sentence justification
Note: Score 5 requires both faithfulness AND complete coverage.
```

| Component | Choice |
|-----------|--------|
| Tracing | LangSmith (auto via `LANGCHAIN_TRACING_V2=true`) |
| Evaluator | LangSmith LLM-as-judge (faithfulness, score 1–5) |
| Test document | DC exam notes PDF (APSIT TE CSE-AIML Sem VI) |

Add to your `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=askmeque
```
Get a free key at [smith.langchain.com](https://smith.langchain.com)

### Table extraction
`PyPDFLoader` dumps PDF text as a flat stream — tables come out as garbled rows with no structure. Added `pdfplumber` to run alongside the existing loader: it detects tables per page, converts each to a pipe-delimited text block tagged `[TABLE]`, and adds them as extra chunks into the same FAISS index.

This means questions about data in tables (comparisons, numbers, structured lists) now retrieve the right chunks instead of getting nothing or garbage.

### Code hardening
- **Constants** — `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_K`, `RERANK_TOP_N`, `LLM_MODEL`, `NO_INFO_RESPONSE` all defined once and referenced everywhere. Change a number in one place, the whole pipeline updates.
- **Single source of truth** — `app.py` imports constants from `day4.py` instead of maintaining its own copies. The debug panel labels read `RETRIEVAL_K` and `RERANK_TOP_N` directly.
- **Empty retrieval short-circuit** — if FAISS + Cohere return no docs, the LLM call is skipped entirely. Saves tokens, avoids hallucination on empty context.
- **No silent failures** — `extract_table_chunks` now prints the error instead of swallowing it with bare `except: pass`.
- **Page numbering** — table chunk metadata stores `page_num + 1` so sources display "Page 1" not "Page 0".

---

## Concepts learned

**Day 1**
- RAG — why we retrieve context at query time instead of fine-tuning
- Embeddings — converting text to vectors that encode meaning
- Chunking & overlap — why we split documents and why chunks overlap
- Vector search — how FAISS finds nearest neighbors using distance metrics
- LCEL — composing pipeline steps with the pipe operator

**Day 2**
- Persistence — saving and loading a vector index
- RunnableParallel — running two branches on the same input
- Prompt hardening — strict grounding rules to prevent hallucination
- Retrieval has no "I don't know" — relevance judgment lives in the LLM, not the retriever
- Distance metrics — L2 vs cosine and why the choice matters

**Day 3**
- Conversational RAG — stateful vs stateless interactions
- Query reformulation — resolving pronouns and references before retrieval
- Streamlit architecture — session state, caching, file upload
- Multi-document retrieval — merging FAISS stores across PDFs

**Day 4**
- Graceful degradation — handling API failures without crashing
- Agent consolidation — merging a separate analysis agent into the answer prompt
- Why follow-up questions never need query analysis — reformulation covers it
- Hash-based caching — MD5 fingerprinting for instant PDF reloads

**Day 5**
- Retrieval evaluation — how to measure whether the right chunk is being fetched
- Bi-encoders vs cross-encoders — why reranking is more accurate than embedding similarity alone
- The two-stage retrieval pattern — fetch wide, rerank precise
- Pipeline transparency — making every step of RAG visible and debuggable
- Evaluation without ground truth — using the debug UI to manually verify retrieval quality

**Day 6**
- Tracing — every LangChain call is auto-logged; see exactly what the retriever fetched and what the LLM saw
- LLM-as-judge — a practical way to evaluate faithfulness without manually writing expected answers
- Traceability over accuracy — evaluation without ground truth is possible when you can verify grounding instead
- Table extraction — why flat text loaders lose structure, and how pdfplumber recovers it
- Single source of truth — why duplicating constants across files creates drift bugs
- Short-circuit patterns — skipping expensive LLM calls when retrieval returns nothing
- Silent exceptions are worse than loud ones — bare `except: pass` hides real bugs

---


## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install langchain langchain-core langchain-community langchain-text-splitters \
            langchain-groq langchain-huggingface langchain-cohere \
            sentence-transformers faiss-cpu pypdf pdfplumber python-dotenv streamlit
```

Create a `.env` file:

GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
Get a free Groq key at [console.groq.com](https://console.groq.com).
Get a free Cohere key at [dashboard.cohere.com](https://dashboard.cohere.com).

---

## Run

```bash
# terminal
python day4.py

# web UI
streamlit run app.py
```

---



