from dotenv import load_dotenv
import os
from pathlib import Path
from pypdf import PdfReader
import re
from openai import OpenAI
import numpy as np
import faiss

# Load .env (expects OPENAI_API_KEY)
if load_dotenv():
    print("Loaded openai api key")
else:
    print("no api key loaded check out .env")
    
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for p in reader.pages:
        t = p.extract_text() or ""
        t = re.sub(r"[“”]", '"', t)
        t = re.sub(r"[’]", "'", t)
        t = re.sub(r"[–—]", "-", t)
        t = re.sub(r"\s+", " ", t).strip()
        parts.append(t)
    return "\n".join(parts)

def load_corpus(pdf_dir: Path):
    corpus = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        text = read_pdf_text(pdf)
        corpus.append({"doc_id": pdf.name, "text": text})
    return corpus

DATA_DIR = Path("./brightleaf_pdfs")  # <- your BrightLeaf PDF folder
assert DATA_DIR.exists(), f"{DATA_DIR} not found. Put PDFs there."

corpus = load_corpus(DATA_DIR)
print(f"Loaded {len(corpus)} docs:", [c["doc_id"] for c in corpus])

def simple_chunks(text: str, chunk_size=900, overlap=150):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i+chunk_size]
        end = chunk.rfind(". ")
        if end > int(chunk_size * 0.6):
            chunk = chunk[:end+1]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        i += max(1, len(chunk) - overlap)
    return chunks

docs = []
for d in corpus:
    for i, ch in enumerate(simple_chunks(d["text"])):
        docs.append({"doc_id": d["doc_id"], "chunk_id": i, "text": ch})
print("Total chunks:", len(docs))

def embed_texts(texts, model="text-embedding-3-small", batch_size=128):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    return np.vstack(vecs)

chunk_texts = [d["text"] for d in docs]
emb = embed_texts(chunk_texts)
dim = emb.shape[1]
print(emb.shape)

faiss.normalize_L2(emb)
index = faiss.IndexFlatIP(dim)
index.add(emb)
print("FAISS index ready. Vectors:", index.ntotal)
id2meta = {i: docs[i] for i in range(len(docs))}

def retrieve(query, k=3, model="text-embedding-3-small"):
    q_emb = embed_texts([query], model=model)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    hits = []
    for score, idx in zip(D[0], I[0]):
        meta = id2meta[int(idx)]
        hits.append({"score": float(score), **meta})
    return hits

for h in retrieve("What is BrightLeaf Solar's mission?", k=3):
    print(round(h["score"], 3), h["doc_id"], f"chunk {h['chunk_id']}", f"text: {h['text']}")
    
def ask_llm(query, contexts, use_rag, model="gpt-4o-mini", temperature=0.2, max_chars=3500):
    ctx = "\n\n---\n\n".join(c["text"] for c in contexts)
    ctx = ctx[:max_chars]
    if use_rag:
        prompt = (
            "Use ONLY the provided context to answer the question.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
        )
    else:
        prompt = query
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

q = "When did BrightLeaf partner with SunSpan and what did they focus on?"
ctx = retrieve(q, k=3)
for c in ctx:
    print("Context score:", c["score"])
    print("Context doc:", c["doc_id"])
    print("Context chunk:", c["chunk_id"])
    print("Context text:", c["text"])
    print()
print(ask_llm(q, ctx, use_rag=False))