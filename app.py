import tempfile
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from day4 import ask, build_sources, embeddings

st.set_page_config(page_title="Chat with PDFs", page_icon="📄", layout="centered")

@st.cache_resource(show_spinner="Building index from PDFs...")
def get_retriever(file_bytes_tuple, filenames_tuple):
    all_chunks  = []
    total_pages = 0

    for file_bytes, filename in zip(file_bytes_tuple, filenames_tuple):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file_bytes)
            tmp_path = f.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages  = loader.load()
            for page in pages:
                page.metadata["source_file"] = filename
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks   = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            total_pages += len(pages)
        finally:
            os.unlink(tmp_path)

    vs = FAISS.from_documents(all_chunks, embeddings)
    return vs.as_retriever(search_kwargs={"k": 4}), total_pages, len(all_chunks)

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