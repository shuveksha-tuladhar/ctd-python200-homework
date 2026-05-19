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

