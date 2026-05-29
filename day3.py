import os
import streamlit as st
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ---- prompts ----
reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the new question as a fully self-contained search query "
     "that does NOT rely on the conversation history. "
     "If already self-contained, return it as-is. "
     "Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant that answers questions about a document.\n"
     "Rules:\n"
     "1. Answer ONLY using the context provided below.\n"
     "2. If the context does not contain the answer, reply exactly: "
     "\"I don't have enough information in the document to answer that.\"\n"
     "3. Do not use outside knowledge. Do not guess.\n"
     "4. Be concise and factual.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ---- core functions (importable) ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question: str, chat_history: list, retriever) -> tuple:
    if chat_history:
        standalone = (reformulate_prompt | llm | StrOutputParser()).invoke({
            "chat_history": chat_history,
            "question": question
        })
    else:
        standalone = question

    docs    = retriever.invoke(standalone)
    context = format_docs(docs)
    answer  = (answer_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question
    })
    return answer, docs

def build_sources(answer: str, docs: list) -> list:
    if "i don't have enough information" in answer.lower():
        return []
    seen, sources = set(), []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        file = doc.metadata.get("source_file", "Unknown")

        key = (file, page)

        if key not in seen:
            seen.add(key)

            snippet = doc.page_content[:100].replace("\n", " ")

            sources.append(
                f"{file} | Page {page}: {snippet}..."
            )

    return sources

# ---- terminal loop (local only) ----
if __name__ == "__main__":
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    PDF_PATHS = ["Test.pdf"]  # Add more PDF paths here
    INDEX_DIR = "faiss_index"

    def build_or_load_vectorstore(pdf_paths):
        if os.path.exists(INDEX_DIR):
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        
        # Process multiple PDFs
        all_chunks = []
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found, skipping...")
                continue
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Add source_file metadata to each document
            for page in pages:
                page.metadata["source_file"] = os.path.basename(pdf_path)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No valid PDFs found!")
        
        vs = FAISS.from_documents(all_chunks, embeddings)
        vs.save_local(INDEX_DIR)
        return vs

    vectorstore = build_or_load_vectorstore(PDF_PATHS)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

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

        answer, docs = ask(question, chat_history, retriever)
        sources      = build_sources(answer, docs)

        print(f"\nANSWER: {answer}")
        if sources:
            print("\nSOURCES:")
            for src in sources:
                print(f"  - {src}")
        print()

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
