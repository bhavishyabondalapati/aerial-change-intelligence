import json

import numpy as np

from aci.rag import build_index, chunk_text, load_index, retrieve

VOCAB = ["drought", "water", "yellow", "nitrogen", "pond", "fish", "blast"]


def fake_embed(texts):
    """Counts a few keywords: enough to test retrieval logic without calling Gemini."""
    return np.array([[t.lower().count(w) for w in VOCAB] + [0.01] for t in texts], np.float32)


def test_chunks_overlap_and_keep_words_whole():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    words = set(text.split())
    assert all(w in words for c in chunks for w in c.split())  # no word was cut in half
    assert chunks[0].split()[-1] in chunks[1]  # consecutive chunks share some text
    assert chunks[-1].endswith("word399")


def test_retrieve_finds_the_chunk_about_the_question():
    chunks = [{"source": "a.pdf", "page": 1, "text": "Fish pond conversion in the delta"},
              {"source": "b.pdf", "page": 7, "text": "Under drought, save water with alternate wetting and drying"},
              {"source": "c.pdf", "page": 3, "text": "Yellow leaves often mean nitrogen deficiency"}]
    vectors = fake_embed([c["text"] for c in chunks])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    hits = retrieve("what to do in a drought with little water", chunks, vectors, fake_embed, k=2)
    assert hits[0]["source"] == "b.pdf" and hits[0]["page"] == 7
    assert hits[0]["score"] > hits[1]["score"]


def test_index_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("aci.rag.read_pdf_pages", lambda p: [(1, "Blast disease of rice. " * 20), (2, "Drought water saving")])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.pdf").write_bytes(b"")
    chunks, vectors = build_index(tmp_path / "docs", tmp_path / "index", fake_embed)
    loaded_chunks, loaded_vectors = load_index(tmp_path / "index")
    assert loaded_chunks == chunks and np.allclose(loaded_vectors, vectors)
    assert {c["page"] for c in chunks} == {1, 2}
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)  # stored unit length, so dot product = cosine
    assert json.loads((tmp_path / "index" / "chunks.json").read_text())[0]["source"] == "guide.pdf"


def test_index_build_resumes_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("aci.rag.read_pdf_pages", lambda p: [(1, "Drought water saving"), (2, "Yellow nitrogen")])
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.pdf").write_bytes(b"")
    calls = []
    counting_embed = lambda texts: calls.append(len(texts)) or fake_embed(texts)
    build_index(tmp_path / "docs", tmp_path / "index", counting_embed)
    build_index(tmp_path / "docs", tmp_path / "index", counting_embed)  # second run: everything cached
    assert calls == [2]
