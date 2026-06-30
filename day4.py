import os
import warnings

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---- config ----
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL      = "llama-3.3-70b-versatile"
RETRIEVAL_K    = 8
RERANK_TOP_N   = 4
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
SOURCE_SNIPPET = 100

NO_INFO_RESPONSE = "I don't have enough information in the document to answer that."

# ---- secrets ----
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
try:
    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"]    = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"]    = "chat-with-pdf"
except Exception:
    pass

# ---- models ----
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm        = ChatGroq(model=LLM_MODEL, temperature=0)

# ---- prompts ----
reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are given a conversation history and a follow-up question.\n"
     "Rewrite the follow-up as a fully self-contained search query.\n\n"
     "Rules:\n"
     "1. Preserve the specific topic from the conversation history.\n"
     "2. Replace ALL pronouns (it, its, they, this, that) with the actual topic.\n"
     "3. If the follow-up is vague (e.g. 'types of', 'tell me more', 'examples'),\n"
     "   combine it with the last discussed topic to form a complete query.\n"
     "4. Return ONLY the rewritten query. No explanation.\n\n"
     "Examples:\n"
     "History: discussed replication | Follow-up: 'types of' → 'types of replication'\n"
     "History: discussed mutual exclusion | Follow-up: 'types of' → 'types of mutual exclusion algorithms'\n"
     "History: discussed Suzuki-Kasami | Follow-up: 'how does it work?' → 'how does the Suzuki-Kasami algorithm work?'\n"
     "History: discussed process migration | Follow-up: 'advantages' → 'advantages of process migration'"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     f"""You are a document question-answering assistant.

Rules:

1. Use ONLY the provided context.

2. If the user's question clearly refers to a specific topic and the answer
exists in the context, answer concisely.

3. If the user's question contains an unresolved reference or does not specify
what it is asking about (for example: "What are its types?",
"How does it work?", "What are the advantages?"), ask a brief clarification
question instead of guessing.

4. If the question is clear but the answer is not present in the context,
reply exactly:
"{NO_INFO_RESPONSE}"
5. If the answer is explicitly stated as a number or formula in the context, quote it exactly.
6. Never use outside knowledge.
7. Never invent facts.

Context:
{{context}}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ---- core functions ----
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_reranked_retriever(vectorstore):
    """FAISS fetches RETRIEVAL_K candidates; Cohere cross-encoder reranks to RERANK_TOP_N."""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=RERANK_TOP_N)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )


def ask(question: str, chat_history: list, retriever) -> tuple[str, list[Document], list[Document]]:
    if chat_history:
        standalone = (reformulate_prompt | llm | StrOutputParser()).invoke({
            "chat_history": chat_history,
            "question": question
        })
    else:
        standalone = question

    # Retrieve candidate chunks via FAISS similarity search
    base_docs = retriever.base_retriever.invoke(standalone)

    # Rerank with Cohere cross-encoder for precision
    docs = retriever.invoke(standalone)

    if not docs:
        return NO_INFO_RESPONSE, [], []

    context = format_docs(docs)

    try:
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })
        return answer, docs, base_docs

    except Exception as e:
        if "rate limit" in str(e).lower():
            return "Groq daily token limit reached. Please try again later.", [], []
        raise


def extract_table_chunks(pdf_path: str, filename: str) -> list[Document]:
    """Extract tables from a PDF with pdfplumber and return them as Document chunks."""
    try:
        import pdfplumber
    except ImportError:
        return []

    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    rows = [
                        " | ".join(cell.strip() if cell else "" for cell in row)
                        for row in table
                    ]
                    table_text = "\n".join(rows)
                    if len(table_text.strip()) < 20:
                        continue
                    chunks.append(Document(
                        page_content=f"[TABLE]\n{table_text}",
                        metadata={"page": page_num + 1, "source_file": filename, "type": "table"}
                    ))
    except Exception as e:
        print(f"Table extraction failed for {filename}: {e}")

    return chunks


def build_sources(answer: str, docs: list[Document]) -> list[str]:
    if any(phrase in answer.lower() for phrase in [
        NO_INFO_RESPONSE.lower(),
        "could you clarify",
        "not specific enough",
        "please clarify",
        "did you mean",
    ]):
        return []

    seen, sources = set(), []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        file = doc.metadata.get("source_file", "Unknown")
        key  = (file, page)
        if key not in seen:
            seen.add(key)
            snippet = doc.page_content[:SOURCE_SNIPPET].replace("\n", " ")
            sources.append(f"{file} | Page {page}: {snippet}...")

    return sources


# ---- terminal loop (local only) ----
if __name__ == "__main__":
    PDF_PATHS = ["NOTES.pdf"]
    INDEX_DIR = "faiss_index"

    def build_or_load_vectorstore(pdf_paths):
        if os.path.exists(INDEX_DIR):
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

        all_chunks = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found, skipping...")
                continue

            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()
            for page in pages:
                page.metadata["source_file"] = os.path.basename(pdf_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(pages)
            chunks += extract_table_chunks(pdf_path, os.path.basename(pdf_path))
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No valid PDFs found!")

        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)
        return vectorstore

    vectorstore = build_or_load_vectorstore(PDF_PATHS)
    retriever   = get_reranked_retriever(vectorstore)

    print("\nReady. Ask questions (type 'quit' to exit, 'clear' to reset memory).\n")
    chat_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "clear":
            chat_history = []
            print("Memory cleared.\n")
            continue
        if not question:
            continue

        answer, docs, base_docs = ask(question, chat_history, retriever)
        sources = build_sources(answer, docs) if docs else []

        print(f"\nANSWER: {answer}")
        if sources:
            print("\nSOURCES:")
            for src in sources:
                print(f"  - {src}")
        print()

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
