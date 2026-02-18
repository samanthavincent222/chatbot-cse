from pypdf import PdfReader
import numpy as np
from groq import Groq

# Optional: sentence-transformers for embeddings
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# ---------- PDF loading ----------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        ptxt = page.extract_text() or ""
        text += ptxt + "\n"
    return text

pdf_text = load_pdf_text("cse_notes.pdf")
print("PDF loaded. Characters:", len(pdf_text))

# ---------- Chunking ----------
def chunk_text(text, chunk_size=1000, overlap=200):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

chunks = chunk_text(pdf_text, chunk_size=800, overlap=100)
print("Created", len(chunks), "chunks from PDF.")

# ---------- Embeddings & in-memory vector DB ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    # returns numpy array shape (n, dim)
    return embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

chunk_embeddings = embed_texts(chunks)
# normalize for faster cosine-similarity via dot product
emb_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
chunk_embeddings_normalized = chunk_embeddings / np.clip(emb_norms, 1e-10, None)

def retrieve(query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    sims = chunk_embeddings_normalized.dot(q_emb_norm)
    idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in idx], sims[idx]

# ---------- Groq client ----------
client = Groq()

# ---------- Topic Filter ----------
ALLOWED_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "python", "java", "c", "c++", "javascript",
    "react", "node", "express", "django", "flask",
    "html", "css", "frontend", "backend",
    "database", "sql", "mongodb",
    "algorithm", "data structure",
    "operating system", "computer networks", "dbms",
    "system design", "compiler", "software engineering",
    "full stack", "cse", "computer science"
]

print("AI Bot 🤖 | AI / Full-Stack / CSE only | type 'bye' to exit\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "bye":
        print("Bot: Bye 👋")
        break

    # ---------- Python-side restriction ----------
    if not any(keyword in user_input.lower() for keyword in ALLOWED_KEYWORDS):
        print(
            "Bot: I can only help with AI, Full-Stack Development, and CSE-related questions.\n"
        )
        continue

    # ---------- Retrieval ----------
    retrieved_passages, scores = retrieve(user_input, top_k=3)
    context = "\n\n---\n\n".join(retrieved_passages)

    # ---------- AI-side restriction + RAG prompt ----------
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\n\n"
        "Use only the provided context to answer when relevant. If the context does not contain the answer, answer based on your knowledge but say you couldn't find exact references in the provided docs."
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict technical assistant.\n"
                    "ONLY answer questions related to:\n"
                    "- Artificial Intelligence\n"
                    "- Machine Learning / Deep Learning\n"
                    "- Full-Stack Development\n"
                    "- Computer Science & Engineering subjects\n"
                    "- Programming, algorithms, databases, system design\n\n"
                    "If a question is outside these topics, reply exactly:\n"
                    "'I can only help with AI, Full-Stack Development, and CSE-related questions.'"
                )
            },
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )

    print("Bot: ", end="", flush=True)
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

    print("\n")
