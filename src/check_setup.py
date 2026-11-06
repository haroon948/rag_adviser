import ollama
from sentence_transformers import SentenceTransformer

reply = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Reply with the word ready."}],
)
print("LLM:", reply.message.content)

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # open model, Apache 2.0 licence
vectors = embedder.encode(["soil nitrogen availability"])
print("Embedding shape:", vectors.shape)  # expect (1, 384)