# eval_simple.py
from day4 import ask, embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# LLM-based faithfulness checker
faithfulness_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge.\n"
     "Given a context and an answer, determine if the answer is fully supported by the context.\n"
     'Return ONLY valid JSON: {{"score": 1}} if faithful, {{"score": 0}} if not.\n'
     "Do not add explanation."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])

relevancy_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an evaluation judge.\n"
     "Given a question and an answer, score how relevant the answer is to the question.\n"
     'Return ONLY valid JSON: {{"score": <0.0 to 1.0>}}\n'
     "Do not add explanation."),
    ("human", "Question: {question}\n\nAnswer: {answer}")
])
def score_faithfulness(context: str, answer: str) -> float:
    try:
        raw = (faithfulness_prompt | llm | StrOutputParser()).invoke({
            "context": context, "answer": answer
        })
        print(f"[DEBUG faithfulness raw]: {raw[:200]}")  # ← add this
        result = json.loads(raw.strip().replace("```json","").replace("```",""))
        return float(result.get("score", 0))
    except Exception as e:
        print(f"[DEBUG faithfulness error]: {e}")  # ← add this
        return 0.0

def score_relevancy(question: str, answer: str) -> float:
    try:
        raw = (relevancy_prompt | llm | StrOutputParser()).invoke({
            "question": question, "answer": answer
        })
        print(f"[DEBUG relevancy raw]: {raw[:200]}")  # ← add this
        result = json.loads(raw.strip().replace("```json","").replace("```",""))
        return float(result.get("score", 0))
    except Exception as e:
        print(f"[DEBUG relevancy error]: {e}")  # ← add this
        return 0.0

# ---- test set ----
test_set = [
    {"question": "What is replication?",
     "ground_truth": "keeping multiple copies of data on different machines"},
    {"question": "What is the Suzuki-Kasami algorithm?",
     "ground_truth": "token-based mutual exclusion algorithm with one circulating token"},
    {"question": "What is process migration?",
     "ground_truth": "relocating a running process from one node to another"},
    {"question": "What is mutual exclusion?",
     "ground_truth": "ensuring only one process accesses a critical section at a time"},
    {"question": "What are the three phases of Suzuki-Kasami?",
     "ground_truth": "requesting CS, receiving request, releasing CS"},
    # add more here
]

# ---- build retriever ----
INDEX_DIR = "faiss_index"
vectorstore = FAISS.load_local(
    INDEX_DIR, embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---- run evaluation ----
print("\nRunning evaluation...\n")
results = []

for item in test_set:
    q  = item["question"]
    gt = item["ground_truth"]

    answer, docs = ask(q, [], retriever)
    context      = "\n\n".join(doc.page_content for doc in docs)

    faith   = score_faithfulness(context, answer)
    rel     = score_relevancy(q, answer)

    # Simple ground truth match — does the answer contain key phrases?
    gt_match = 1.0 if any(
        word.lower() in answer.lower()
        for word in gt.split()
        if len(word) > 4   # skip short words
    ) else 0.0

    results.append({
        "question":    q,
        "answer":      answer[:80] + "...",
        "faithfulness": faith,
        "relevancy":   rel,
        "gt_match":    gt_match,
    })
    print(f"✓ {q[:50]}")

# ---- print report ----
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print(f"{'Question':<45} {'Faith':>6} {'Rel':>6} {'GT':>6}")
print("-"*60)
for r in results:
    print(f"{r['question'][:44]:<45} {r['faithfulness']:>6.2f} {r['relevancy']:>6.2f} {r['gt_match']:>6.2f}")

print("-"*60)
avg_faith = sum(r["faithfulness"] for r in results) / len(results)
avg_rel   = sum(r["relevancy"]    for r in results) / len(results)
avg_gt    = sum(r["gt_match"]     for r in results) / len(results)
print(f"{'MEAN':<45} {avg_faith:>6.2f} {avg_rel:>6.2f} {avg_gt:>6.2f}")
print("="*60)