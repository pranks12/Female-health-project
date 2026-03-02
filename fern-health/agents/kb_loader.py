import json
import os
import re
from pathlib import Path


KB_DIR = Path(__file__).parent.parent / "kb" / "agent3_treatment"

_kb_store: dict = {}  # topic -> full document
_kb_index: list = []  # flat list of {topic, condition/section, keywords, text}


def _extract_text(obj, path="") -> list[dict]:
    """Recursively extract searchable text chunks from a KB document."""
    chunks = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            chunks.extend(_extract_text(val, path=f"{path}.{key}" if path else key))
    elif isinstance(obj, list):
        for item in obj:
            chunks.extend(_extract_text(item, path=path))
    elif isinstance(obj, str) and len(obj) > 20:
        chunks.append({"path": path, "text": obj})
    return chunks


def load_knowledge_base() -> dict:
    """Load all JSON files from the KB directory into memory."""
    global _kb_store, _kb_index
    _kb_store = {}
    _kb_index = []

    if not KB_DIR.exists():
        return {}

    for file in KB_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            topic = data.get("topic", file.stem)
            _kb_store[file.stem] = data

            chunks = _extract_text(data)
            _kb_index.append({
                "file": file.stem,
                "topic": topic,
                "chunks": chunks,
                "raw": data,
            })
        except Exception as e:
            print(f"Warning: Could not load {file.name}: {e}")

    print(f"Knowledge base loaded: {len(_kb_store)} files, {sum(len(e['chunks']) for e in _kb_index)} text chunks")
    return _kb_store


def _score_entry(entry: dict, query_terms: list) -> float:
    """Score a KB entry by how many query terms appear in its text chunks."""
    score = 0.0
    topic = entry["topic"].lower()
    file = entry["file"].lower()
    full_text = " ".join(c["text"].lower() for c in entry["chunks"])
    full_text += f" {topic} {file}"

    for term in query_terms:
        term = term.lower()
        if term in topic or term in file:
            score += 3.0
        count = full_text.count(term)
        if count > 0:
            score += min(count * 0.5, 3.0)

    return score


def retrieve(query: str, top_k: int = 3) -> list:
    """
    Retrieve the most relevant KB documents for a query.
    Returns a list of the top_k matching documents (raw JSON).
    """
    if not _kb_index:
        load_knowledge_base()

    if not _kb_index:
        return []

    stopwords = {"what", "how", "why", "is", "are", "the", "a", "an", "for", "to",
                 "of", "i", "do", "have", "can", "with", "about", "me", "my", "and"}
    terms = [t for t in re.split(r'\W+', query.lower()) if t and t not in stopwords and len(t) > 2]

    if not terms:
        return [e["raw"] for e in _kb_index[:top_k]]

    scored = [(e, _score_entry(e, terms)) for e in _kb_index]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = [e["raw"] for e, score in scored if score > 0]
    if not results:
        results = [e["raw"] for e, _ in scored]

    return results[:top_k]


def get_all_topics() -> list:
    """Return list of all loaded KB topics."""
    return [entry["topic"] for entry in _kb_index]


def get_kb_summary() -> dict:
    """Return summary stats about the loaded knowledge base."""
    return {
        "loaded": len(_kb_store) > 0,
        "files": len(_kb_store),
        "topics": get_all_topics(),
        "total_chunks": sum(len(e["chunks"]) for e in _kb_index),
    }
