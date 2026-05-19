# Part 2: Mini-Project: Groundwork Coffee Co. Q&A Assistant

from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Step 1: Setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
docs_dir = Path("./resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"\nNumber of documents loaded: {len(documents)}\n")

print("Files loaded:")
for doc in documents:
    file_name = doc.metadata.get("file_name", "Unknown file")
    print(f"- {file_name}")
    
# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")

# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")

    response = query_engine.query(question)

    print("\nANSWER:")
    print(response.response)

    print("\nTOP RETRIEVED SOURCE:")

    if response.source_nodes:
        top_node = response.source_nodes[0]

        file_name = top_node.node.metadata.get("file_name", "Unknown file")
        score = top_node.score
        text_snippet = top_node.node.text[:200].replace("\n", " ")

        print(f"Document: {file_name}")
        print(f"Similarity score: {score}")
        print(f"Text snippet: {text_snippet}")
    else:
        print("No source nodes returned.")
        
# Reflection:
# Overall, the assistant performed very well and stayed grounded in the retrieved documents.
# All five answers were accurate and clearly aligned with the source text chunks returned by the retriever.
# The system showed strong consistency between questions, retrieved context, and final answers. For example, hours, loyalty program rules, and catering/wholesale information were all correctly answered using highly relevant source nodes (with similarity scores mostly in the 0.76–0.90 range).
# One notable observation is that the answers were sometimes slightly more complete than the retrieved snippet itself (e.g., the “How did Groundwork start?” response included extra narrative detail). This suggests the model is effectively synthesizing across retrieved context rather than just copying text.
# Importantly, there were no clear hallucinations. Every response could be traced back to the provided documents. The strongest retrievals (around 0.90+) produced the most precise and confident answers.
# Overall, the RAG pipeline is working as intended: retrieval is accurate, and generation remains faithful to the grounded context.

# Step 5: Find a Failure

hard_question = "What is the refund policy for Groundwork Coffee?"

print(f"QUESTION: {hard_question}")

response = query_engine.query(hard_question)

print("\nANSWER:")
print(response.response)

print("\nTOP 3 RETRIEVED SOURCES:")

for i, node in enumerate(response.source_nodes[:3]):
    file_name = node.node.metadata.get("file_name", "Unknown file")
    score = node.score
    text_snippet = node.node.text[:200].replace("\n", " ")

    print(f"\nSource {i+1}:")
    print(f"Document: {file_name}")
    print(f"Similarity score: {score}")
    print(f"Text snippet: {text_snippet}")

# comment:
# I asked about the refund policy because it is not mentioned in the documents, so I expected the system to struggle or not find a clear answer.
# The assistant correctly said there is no refund policy in the provided context, which shows it handled the missing information well instead of guessing.
# However, the retrieved sources were not very relevant (story, menu, wholesale), which shows the search did not find anything directly related to refunds.
# The answer still sounded confident and clear, even though the retrieval was weak.
# This shows that even when retrieval is not strong, the model can still produce a reasonable-sounding response.
# To improve the system, I would add more specific documents (like policies/FAQ) and improve retrieval so irrelevant chunks are not ranked so highly.

# Step 6: Reflection

# In this project, the LlamaIndex implementation took only a few lines of code (mainly creating the index and query engine). 
# The manual RAG approach from the lesson would have taken many more lines for chunking, embedding, storing vectors, and searching. 
# This shows that frameworks like LlamaIndex save a lot of time and reduce complexity by handling the hard parts for you.
# A different use case for this system could be in a hospital setting, where it answers questions from medical guidelines, patient care instructions, and hospital policies. Staff could quickly find accurate information without reading long documents.
# One failure mode of RAG is that even when the correct information is retrieved, the model can still misinterpret it or combine it incorrectly. This means it can give a confident but slightly wrong answer even with good retrieval.

# --- Optional Extensions ---
# --- Extension A: Side-by-Side Comparison (Moderate) ---

docs_dir = Path("./resources/groundwork_docs")
documents_raw = {f.name: f.read_text() for f in docs_dir.glob("*.txt")}

#  Keyword retrieval helper
def simple_keyword_retrieval(query, documents):
    query_words = set(query.lower().split())

    best_doc = None
    best_score = 0
    best_text = ""

    for name, text in documents.items():
        text_lower = text.lower()
        score = sum(1 for word in query_words if word in text_lower)

        if score > best_score:
            best_score = score
            best_doc = name
            best_text = text

    return best_doc, best_score, best_text[:500]

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print("\n" + "=" * 90)
    print(f"QUESTION: {q}")

    # Semantic RAG (LlamaIndex)
    semantic_response = query_engine.query(q)

    # Keyword RAG
    kw_doc, kw_score, kw_text = simple_keyword_retrieval(q, documents_raw)

    print("\n--- SEMANTIC RAG (LlamaIndex) ---")
    print(semantic_response.response)

    print("\n--- KEYWORD RAG ---")
    print(f"Document: {kw_doc}")
    print(f"Score: {kw_score}")
    print(f"Text snippet: {kw_text[:200]}")

    print("\n--- COMPARISON NOTE ---")
    print("Semantic RAG usually gives better understanding of meaning.")
    print("Keyword RAG depends on exact word matches, so it can miss context.")

# Comparison Reflection:
# Keyword RAG worked well when the question used exact words from the documents, such as hours or loyalty program terms.
# However, it struggled when wording was different or when meaning had to be understood.
# Semantic RAG performed better overall because it understands context, not just keywords.
# In a few simple cases, keyword RAG performed similarly, especially when the answer contained direct matching words.
# Overall, semantic RAG was more reliable and accurate.

# --- Extension C: Add a New Document (Low) ---

# Rebuild index after adding new document
documents = SimpleDirectoryReader(str("./resources/groundwork_docs")).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex rebuilt successfully with new document included.")

test_query = "Is there any upcoming event at Groundwork Coffee?"

print("\nQUESTION:", test_query)

response = query_engine.query(test_query)

print("\nANSWER:")
print(response.response)

print("\nTOP SOURCE:")
if response.source_nodes:
    top = response.source_nodes[0]
    print("Document:", top.node.metadata.get("file_name"))
    print("Score:", top.score)
    print("Snippet:", top.node.text[:200])
    
# Reflection:
# I added a new document called "seasonal_event.txt" which describes a summer community event with free coffee tastings, music, and discounts.
# I tested it using the question: "Is there any upcoming event at Groundwork Coffee?"
# The assistant correctly retrieved the new document and included event details in its answer.
# This shows that RAG systems can be updated easily by adding new documents and rebuilding the index, without retraining the model.
# This is a big advantage over fine-tuning, which would require retraining the model every time new information is added.