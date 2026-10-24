"""Retrieval for the report: split the agriculture guides into chunks, embed them, and find the chunks closest to a question.

Retrieval is plain NumPy (cosine similarity over a few hundred vectors), so no vector database is needed.
The embedding function is passed in, so tests can use a fake one and never call the API.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768  # Gemini embeddings can be shortened; 768 is plenty for a few hundred chunks


def read_pdf_pages(path):
    """[(page_number, text)] for pages that have real text (scanned/photo-only pages are skipped)."""
    from pypdf import PdfReader

    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if len(text) >= 50:
            pages.append((i, text))
    return pages


def chunk_text(text, size=1200, overlap=200):
    """Split text into ~size-character pieces that overlap, cutting at a space so words stay whole."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        nxt = text.find(" ", end - overlap, end)  # next chunk starts at a word boundary inside the overlap
        start = nxt + 1 if nxt > start else end
    return [c for c in chunks if c]


def make_chunks(docs_dir, size=1200, overlap=200):
    """Every chunk keeps its source file and page so the report can cite it."""
    chunks = []
    for pdf in sorted(Path(docs_dir).glob("*.pdf")):
        for page, text in read_pdf_pages(pdf):
            for piece in chunk_text(text, size, overlap):
                chunks.append({"source": pdf.name, "page": page, "text": piece})
    return chunks


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def gemini_embedder(task_type, api_key=None, batch=20, retries=8):
    """Returns embed(list_of_texts) -> (n, EMBED_DIM) array using the Gemini API.

    task_type is RETRIEVAL_DOCUMENT for guide chunks and RETRIEVAL_QUERY for questions: Gemini embeds them
    slightly differently so questions land near the passages that answer them.
    The free tier limits tokens per minute, so chunks go in small batches and a 429 reply means "wait, then retry".
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM)

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            for attempt in range(retries):
                try:
                    res = client.models.embed_content(model=EMBED_MODEL, contents=texts[i : i + batch], config=config)
                    out.extend(e.values for e in res.embeddings)
                    break
                except Exception as e:  # free tier rate limit: wait and retry
                    if "PerDay" in str(e):  # daily quota (each text counts as one request): waiting minutes won't help
                        raise RuntimeError("Daily Gemini embedding quota used up. Progress is cached; "
                                           "run the same command again after the quota resets (about 24 h).") from e
                    if attempt == retries - 1 or "429" not in str(e):
                        raise
                    time.sleep(30 * (attempt + 1))
        return np.array(out, dtype=np.float32)

    return embed


def build_index(docs_dir, index_dir, embed, save_every=20):
    """Embed every chunk and save the index. Embeddings are cached by text hash after each batch,
    so a run stopped by rate limits resumes where it left off, and unchanged chunks are never re-embedded."""
    chunks = make_chunks(docs_dir)
    if not chunks:
        raise FileNotFoundError(f"No PDF text found in {docs_dir}")
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "embedding_cache.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    keys = [hashlib.sha1(c["text"].encode()).hexdigest() for c in chunks]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    for start in range(0, len(todo), save_every):
        part = todo[start : start + save_every]
        for i, vec in zip(part, embed([chunks[i]["text"] for i in part])):
            cache[keys[i]] = vec
        np.savez(cache_path, **cache)
        print(f"  embedded {len(chunks) - len(todo) + start + len(part)}/{len(chunks)}", end="\r", flush=True)
    vectors = normalize_rows([cache[k] for k in keys])
    np.save(out / "vectors.npy", vectors)
    (out / "chunks.json").write_text(json.dumps(chunks, indent=1))
    return chunks, vectors


def load_index(index_dir):
    d = Path(index_dir)
    return json.loads((d / "chunks.json").read_text()), np.load(d / "vectors.npy")


def retrieve(query, chunks, vectors, embed, k=5):
    """The k chunks most similar in meaning to the query, best first, each with its similarity score (-1..1)."""
    q = normalize_rows(embed([query]))[0]
    sims = vectors @ q  # rows are unit length, so a dot product is the cosine similarity
    best = np.argsort(-sims)[:k]
    return [dict(chunks[i], score=float(sims[i])) for i in best]


def main():
    from dotenv import load_dotenv

    load_dotenv(".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--query", help="try a question against an existing index")
    args = ap.parse_args()
    if args.query:
        chunks, vectors = load_index(args.index)
        for hit in retrieve(args.query, chunks, vectors, gemini_embedder("RETRIEVAL_QUERY")):
            print(f"{hit['score']:.3f}  {hit['source']} p.{hit['page']}: {hit['text'][:160]}...")
        return
    chunks, _ = build_index(args.docs, args.index, gemini_embedder("RETRIEVAL_DOCUMENT"))
    print(f"indexed {len(chunks)} chunks from {len({c['source'] for c in chunks})} PDFs -> {args.index}")


if __name__ == "__main__":
    main()
