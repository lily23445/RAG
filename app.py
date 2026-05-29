import tempfile
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from day3 import ask, build_sources, embeddings

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")

# ---- cache the index build so it only runs once per uploaded file ----
@st.cache_resource(show_spinner="Building index from PDF...")
def get_retriever(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp_path = f.name
    loader   = PyPDFLoader(tmp_path)
    pages    = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(pages)
    vs       = FAISS.from_documents(chunks, embeddings)
    os.unlink(tmp_path)
    return vs.as_retriever(search_kwargs={"k": 4}), len(pages), len(chunks)

# ---- session state ----
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "messages"      not in st.session_state: st.session_state.messages      = []
if "retriever"     not in st.session_state: st.session_state.retriever     = None

# ---- sidebar ----
st.title("📄 Chat with PDF")
st.caption("Upload a PDF and ask questions about it.")

with st.sidebar:
    st.header("Upload your PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded:
        retriever, n_pages, n_chunks = get_retriever(
            uploaded.read(), uploaded.name
        )
        st.session_state.retriever = retriever
        st.success(f"✅ {uploaded.name}")
        st.caption(f"{n_pages} pages · {n_chunks} chunks")

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
if question := st.chat_input("Ask a question about your PDF..."):
    if st.session_state.retriever is None:
        st.warning("Please upload a PDF first.")
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