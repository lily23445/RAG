# retrieval_debug.py
from day4 import embeddings,ask
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "faiss_index", embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Test questions with expected keywords that should appear in retrieved chunks
test_set = [
    # Factual / direct
    {
        "question": "How many messages does Ricart-Agrawala use per CS entry?",
        "ground_truth": "2(N-1) messages"
    },
    {
        "question": "What is the default block size in HDFS?",
        "ground_truth": "128 MB"
    },
    {
        "question": "What is the message complexity of the Bully election algorithm?",
        "ground_truth": "O(N squared) in worst case"
    },
    {
        "question": "How many messages does the Ring election algorithm use?",
        "ground_truth": "2N messages"
    },
    {
        "question": "What does LN stand for in the Suzuki-Kasami token?",
        "ground_truth": "sequence number of the last CS execution by each process"
    },

    # List-based (hallucination-prone)
    {
        "question": "What are the four mutual exclusion algorithms mentioned?",
        "ground_truth": "Lamport, Ricart-Agrawala, Maekawa, Suzuki-Kasami"
    },
    {
        "question": "What are the three components of a process to be migrated?",
        "ground_truth": "code segment, resource segment, execution segment"
    },
    {
        "question": "What are the four client-centric consistency models?",
        "ground_truth": "monotonic reads, monotonic writes, read your writes, writes follow reads"
    },
    {
        "question": "What are the five failure models from easiest to hardest?",
        "ground_truth": "crash, omission, timing, response, byzantine"
    },

    # Comparison
    {
        "question": "What is the difference between load balancing and load sharing?",
        "ground_truth": "load balancing equalizes load continuously, load sharing just prevents idle nodes"
    },
    {
        "question": "What is the difference between weak mobility and strong mobility?",
        "ground_truth": "weak moves only code, strong moves code plus execution state"
    },

    # Multi-hop
    {
        "question": "Why does Ricart-Agrawala use fewer messages than Lamport?",
        "ground_truth": "deferred REPLY acts as implicit RELEASE so no separate RELEASE message needed"
    },
    {
        "question": "What makes Suzuki-Kasami token-based and what is its message complexity?",
        "ground_truth": "unique token circulates, only token holder enters CS, 0 or N messages"
    },

    # Out of scope
    {
        "question": "What is the CAP theorem?",
        "ground_truth": "not in document"
    },
    {
        "question": "What is MapReduce?",
        "ground_truth": "not in document"
    },
]

print("\nRETRIEVAL DIAGNOSTIC")
print("="*60)

for item in test_set:
    q        = item["question"]
    gt       = item["ground_truth"]
    docs     = retriever.invoke(q)

    # check if ground truth keywords appear in retrieved chunks
    combined = " ".join(doc.page_content for doc in docs).lower()
    gt_words = [w.strip('.,()') for w in gt.split() if len(w) > 2]
    hits     = [w for w in gt_words if w.lower() in combined]
    score    = len(hits) / len(gt_words) if gt_words else 0

    status = "✅" if score >= 0.5 else "❌"
    pages  = [d.metadata.get("page") for d in docs]

    # in your diagnostic loop, add this check:
    if item["ground_truth"] == "not in document":
        # for out-of-scope questions, check that the CHATBOT refuses
        answer, _ = ask(q, [], retriever)
        refused = "don't have enough information" in answer.lower()
        status  = "✅" if refused else "❌"
        print(f"\n{status} {q[:55]}")
        print(f"   [OUT OF SCOPE] Refused correctly: {refused}")
        print(f"   Answer: {answer[:80]}")
        continue

    print(f"\n{status} {q[:55]}")
    print(f"   Score: {score:.0%} ({len(hits)}/{len(gt_words)} keywords found)")
    print(f"   Pages retrieved: {pages}")
    if score < 0.5:
        missing = [w for w in gt_words if w.lower() not in combined]
        print(f"   Missing keywords: {missing}")

    