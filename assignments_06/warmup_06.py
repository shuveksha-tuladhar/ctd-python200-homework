from dotenv import load_dotenv
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
# --- RAG Concepts ---

# Concepts Q1

# Scenario A:
# Best approach: RAG (Retrieval-Augmented Generation)
# The legal team's documents change oftsen, so RAG is the best choice because it can pull updated information from the PDF library without retraining the model. It also works well for searching through many documents and answering questions accurately.

# Scenario B:
# Best approach: Fine-tuning
# The startup wants the model to consistently write in a unique brand voice. Since they already have 3,000 examples of past writing, fine-tuning can train the model to match that specific style closely.

# Scenario C:
# Best approach: Prompt engineering
# The analyst only needs answers from one short report and does not need a reusable system. Simply placing the report into the prompt is fast, simple, and effective without needing RAG or fine-tuning.

# Concepts Q2

# A confidently wrong answer is more harmful because people are more likely to trust and act on it. When a model sounds certain, users may not question the information even if it is incorrect.
# Example: In healthcare, an AI assistant could confidently give the wrong medication dosage to a doctor or patient. This could lead to serious injury or even death.
# The tone matters because confidence affects trust. If the model says "I am not sure," users are more careful and may double check the information. A confident tone can make false information seem reliable.


# Concepts Q3

# Correct order of a RAG pipeline:

steps = [
    "Receive the user's query",              # The system gets the question from the user.
    
    "Extract text from source documents",   # Text is collected from PDFs, files, or databases.
    
    "Split text into chunks",               # Large documents are broken into smaller sections for easier searching.
    
    "Convert text chunks into embeddings",  # Each chunk is turned into a numerical vector representation.
    
    "Embed the user's query",               # The user's question is also converted into an embedding.
    
    "Retrieve the most relevant chunks",    # The system finds the chunks most similar to the user's query.
    
    "Inject retrieved chunks into the prompt",  # The relevant chunks are added into the prompt sent to the LLM.
    
    "Generate a response from the LLM"      # The model creates a final answer using the provided context.
]

# --- Keyword RAG ---

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keyword Q1 
print("\n--- Keyword Q1 ---")

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

selected_doc = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", selected_doc)

# Result comment:
# The selected document was "loyalty.txt". Keyword retrieval got this wrong. The correct answer should have been "hours.txt", but keyword retrieval only matches exact words or tokens.
# The query included the word "your", and "loyalty.txt" contains the phrase "your choice", which created a stronger keyword match than the actual  hours document.
# Also, the query used the word "weekend" while the document used "weekends", so the simple keyword matcher may not have recognized them as the same word.
# This shows a limitation of basic keyword retrieval because it does not truly understand meaning or context.

# Keyword Q2
print("\n--- Keyword Q2 ---")

query = "Do you have anything without caffeine?"
results = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", results)

# The selected document was "None found".
# Keyword retrieval got this wrong because none of the documents contain the exact words "caffeine" or "without".
# This shows a limitation of keyword-based RAG because it only matches exact words and cannot understand concepts like "caffeine-free" or "decaf".
# Keyword RAG is not able to infer meaning, so it misses relevant documents even when they are clearly related.
# A semantic or embedding-based retrieval system would perform better because it can match based on meaning instead of exact keyword overlap.

# Keyword Q3
print("\n--- Keyword Q3 ---")

query = "How do I sign up for rewards?"
results3 = simple_keyword_retrieval(query, documents, verbose=True)

print("Selected document:", results3)

# Prediction (before running):
# I predict the selected document will be "loyalty.txt" because the query "How do I sign up for rewards?" is related to rewards, and loyalty.txt contains information about a rewards/points system.

# After running:
# The actual result was "None found".
# My prediction was incorrect.
# Keyword RAG only matches exact tokens, so it failed because none of the documents contain words like "rewards", "sign", or "up".
# Even though "loyalty.txt" is the correct semantic match, keyword retrieval cannot understand that "rewards" is related to a "loyalty program".
# This shows a limitation of keyword-based retrieval, where meaning is ignored and only exact word overlap is used for matching.

# --- Semantic RAG Concepts ---
# Semantic Q1
# What is a vector embedding?
# A vector embedding is a way to turn text into a list of numbers that represents its meaning.
# Texts with similar meanings end up with similar number patterns, even if the words are different.

# Cosine similarity question:
# The chunk with similarity 0.85 is more relevant than 0.30.
# This number tells how close two embeddings are in meaning. A higher score means the texts are more similar in meaning.

# Why semantic search works without exact words:
# Semantic search works because it compares meaning instead of exact words.
# Even if the query and chunk use different words, embeddings can still place them close together in vector space if they mean similar things.

# Semantic Q2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Meaning of text (embeddings) |
# | What is retrieved?         | Full document                     | Relevant chunks of documents |
# | Can it handle synonyms?    | No                                | Yes |
# | Storage format             | Plain text dictionary             | Vector embeddings in a vector database |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity between vectors |
