#goal is to add Persistence — stop re-embedding the PDF on every run (save the FAISS index to disk)
#Source citations — show which pages each answer came from
#hardened prompt — make it refuse to answer when the PDF doesn't cover something (kills hallucination)

import os
import warnings
from dotenv import load_dotenv #THIS IS TO GET THE .ENV FILES TOKEN
from langchain_community.document_loaders import PyPDFLoader #TO LOAD PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter #TO SPLIT THE DATA INTO CHUNK OF TEXTS
from langchain_huggingface import HuggingFaceEmbeddings # MODEL FOR EMBEDDING
from langchain_groq import ChatGroq #FOR LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS #INDEXING THE EMBEDDED VECTOR AND LINK BETWEEN THOSE VECTORS SHOWING SIMILARITES
from langchain_core.output_parsers import StrOutputParser # OUTPUT BY LLM IS OBJECT SO TO CONVERT IT INTO STRING
from langchain_core.runnables import RunnablePassthrough,RunnableParallel# FOR INPUT TO REMAIN SAME WHEN SENDING TO THE LLM AND THE OTHER FOR MULTIPLE TASK TO RUN PARALLELY

load_dotenv()


PDF_PATH="Test.pdf"
INDEX_DIR = "faiss_index"   

#EMBEDDING 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# BUILD OR LOAD the vector store  (the "persistence" upgrade)
def build_or_load_vectorstore():
    if os.path.exists(INDEX_DIR):
        print("Loading existing FAISS index from disk...")
        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    
    print("No saved index found. Building from PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    #SPLIT
    splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200 )
    chunks=splitter.split_documents(pages)
    print(f"split into {len(chunks)} chunks")

#embed andd store

    vs=FAISS.from_documents(chunks,embeddings)
    vs.save_local(INDEX_DIR)
    print(f"Index saved to '{INDEX_DIR}/'")
    return vs

vectorstore = build_or_load_vectorstore()
retriever=vectorstore.as_retriever(search_kwargs={"k":4})


# ---------------------------------------------------------------------------
# PROMPT (hardened — strict grounding, admits when it doesn't know)
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant that answers questions about a document.\n"
     "Rules:\n"
     "1. Answer ONLY using the context provided below.\n"
     "2. If the context does not contain the answer, reply exactly: "
     "\"I don't have enough information in the document to answer that.\"\n"
     "3. Do not use outside knowledge. Do not guess.\n"
     "4. Be concise and factual.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# THE CHAIN (returns BOTH the answer AND the source documents)

answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain = RunnableParallel(
    answer=answer_chain,
    sources=retriever        # runs retriever again on the same question to grab page metadata
)


# 9. # INTERACTIVE LOOP
if __name__ == "__main__":
    print("\nReady. Ask a question (or type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        result = rag_chain.invoke(question)
        answer = result["answer"]
        print("\nANSWER:", answer)

        refusal = "i don't have enough information"
        if refusal not in answer.lower():
            print("\nSOURCES:")
            seen = set()
            for doc in result["sources"]:
                page = doc.metadata.get("page", "?")
                if page not in seen:
                    seen.add(page)
                    snippet = doc.page_content[:100].replace("\n", " ")
                    print(f"  - Page {page}: {snippet}...")
        print()