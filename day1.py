import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

#load
loader=PyPDFLoader('test.pdf')
pages=loader.load()
print(f"loaded : {len(pages)}  PAGES")

#SPLIT
splitter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200 )
chunks=splitter.split_documents(pages)
print(f"split into {len(chunks)} chunks")

#embed andd store
embeddings =HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=FAISS.from_documents(chunks,embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k":4})

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context below. "
     "If the answer isn't in the context, say you don't know. Be concise.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. ASK
question = "What is the ricart ?"   # change this
answer = rag_chain.invoke(question)
print("\nANSWER:", answer)


# 9. SOURCES
sources = retriever.invoke(question)
print("\nSOURCES:")
for doc in sources:
    print(f"  - Page {doc.metadata.get('page', '?')}: {doc.page_content[:120]}...")
