import tempfile
import os
import json
import hashlib
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from day4 import ask, build_sources, embeddings,get_reranked_retriever

st.set_page_config(page_title="Chat with PDFs", page_icon="📄", layout="centered")


def get_pdf_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:12]

@st.cache_resource(show_spinner=False)
def get_retriever(file_bytes_tuple, filenames_tuple):
    all_chunks      = []
    cached_stores   = []
    total_pages     = 0

    for file_bytes, filename in zip(file_bytes_tuple, filenames_tuple):
        pdf_hash  = get_pdf_hash(file_bytes)
        cache_dir = f"pdf_cache/{pdf_hash}"

        if os.path.exists(cache_dir):
            st.toast(f"⚡ {filename} loaded from cache")
            cached_vs = FAISS.load_local(
                cache_dir, embeddings,
                allow_dangerous_deserialization=True
            )
            cached_stores.append(cached_vs)
            chunks_path = f"{cache_dir}/chunks.json"
            if os.path.exists(chunks_path):
                with open(chunks_path) as f:
                    raw = json.load(f)
                total_pages += len(set(
                    c["metadata"].get("page", 0) for c in raw
                ))
                all_chunks.extend([
                    Document(page_content=c["page_content"], metadata=c["metadata"])
                    for c in raw
                ])
            continue

        with st.spinner(f"Embedding {filename}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(file_bytes)
                tmp_path = f.name
            try:
                loader   = PyPDFLoader(tmp_path)
                pages    = loader.load()
                for page in pages:
                    page.metadata["source_file"] = filename
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                total_pages += len(pages)

                os.makedirs(cache_dir, exist_ok=True)
                vs = FAISS.from_documents(chunks, embeddings)
                vs.save_local(cache_dir)
                cached_stores.append(vs)

                with open(f"{cache_dir}/chunks.json", "w") as f:
                    json.dump([{
                        "page_content": c.page_content,
                        "metadata": c.metadata
                    } for c in chunks], f)
            finally:
                os.unlink(tmp_path)

    # merge all FAISS stores into one
    if len(cached_stores) == 1:
        combined = cached_stores[0]
    else:
        combined = cached_stores[0]
        for store in cached_stores[1:]:
            combined.merge_from(store)

    return get_reranked_retriever(combined), total_pages, len(all_chunks)

# ---- session state ----
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "messages"       not in st.session_state: st.session_state.messages       = []
if "retriever"      not in st.session_state: st.session_state.retriever      = None
if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []

# ---- UI ----
st.title("📄 Chat with Multiple PDFs")
st.caption("Upload one or more PDFs and ask questions about them.")

with st.sidebar:
    st.header("Upload your PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDFs", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        current_filenames = [f.name for f in uploaded_files]

        if current_filenames != st.session_state.uploaded_files:
            st.session_state.uploaded_files = current_filenames
            st.session_state.chat_history   = []
            st.session_state.messages       = []

            file_bytes_list = [f.read() for f in uploaded_files]
            retriever, n_pages, n_chunks = get_retriever(
                tuple(file_bytes_list),
                tuple(current_filenames)
            )
            st.session_state.retriever = retriever
            st.success(f"✅ {len(uploaded_files)} file(s) loaded")
            st.caption(f"{n_pages} pages · {n_chunks} chunks total")

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.messages     = []
        st.rerun()

# ---- chat display ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# ---- chat input ----
if question := st.chat_input("Ask a question about your PDFs..."):
    if st.session_state.retriever is None:
        st.warning("Please upload at least one PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask(
                    question,
                    st.session_state.chat_history,
                    st.session_state.retriever
                )
            st.markdown(answer)
            sources = build_sources(answer, docs)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(src)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })