"""
Knowledge Base Loader for Fern Health agents.

Reads plain-text documents from kb/<agent_name>/, chunks them into
passages, and builds an in-memory store that agents can query at runtime.
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KB_ROOT = Path(__file__).resolve().parent.parent / "kb"
CHUNK_SIZE = 400          # target words per chunk
CHUNK_OVERLAP = 50        # words of overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    agent: str
    source: str           # filename (stem)
    index: int            # chunk number within the document
    text: str

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(agent={self.agent!r}, source={self.source!r}, idx={self.index}, preview={preview!r})"


@dataclass
class KnowledgeBase:
    chunks: list[Chunk] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load_agent(self, agent_name: str) -> int:
        """Load all .txt documents for *agent_name* and return chunks added."""
        agent_dir = KB_ROOT / agent_name
        if not agent_dir.exists():
            print(f"  [WARN] Directory not found: {agent_dir}")
            return 0

        txt_files = sorted(agent_dir.glob("*.txt"))
        if not txt_files:
            print(f"  [WARN] No .txt files found in {agent_dir}")
            return 0

        added = 0
        for path in txt_files:
            added += self._load_file(agent_name, path)
        return added

    def _load_file(self, agent: str, path: Path) -> int:
        """Chunk a single file and append to the store. Returns chunk count."""
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return 0

        words = raw.split()
        chunks_added = 0
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + CHUNK_SIZE]
            text = " ".join(chunk_words)
            self.chunks.append(
                Chunk(agent=agent, source=path.stem, index=i, text=text)
            )
            chunks_added += 1

        return chunks_added

    # ------------------------------------------------------------------ #
    # Retrieval (simple keyword search — no embeddings required)
    # ------------------------------------------------------------------ #

    def search(self, query: str, agent: str | None = None, top_k: int = 3) -> list[Chunk]:
        """Return up to *top_k* chunks most relevant to *query* (BM25-lite)."""
        tokens = set(re.findall(r"\w+", query.lower()))
        pool = [c for c in self.chunks if agent is None or c.agent == agent]

        scored = []
        for chunk in pool:
            chunk_tokens = re.findall(r"\w+", chunk.text.lower())
            score = sum(chunk_tokens.count(t) for t in tokens)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        agents: dict[str, dict] = {}
        for chunk in self.chunks:
            entry = agents.setdefault(chunk.agent, {"documents": set(), "chunks": 0})
            entry["documents"].add(chunk.source)
            entry["chunks"] += 1
        return {
            name: {"documents": len(v["documents"]), "chunks": v["chunks"]}
            for name, v in agents.items()
        }


# ---------------------------------------------------------------------------
# Singleton store (imported by agents)
# ---------------------------------------------------------------------------

_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Return the global KB, initialising it on first call."""
    global _kb
    if _kb is None:
        _kb = _build()
    return _kb


def _build() -> KnowledgeBase:
    kb = KnowledgeBase()
    agent_dirs = sorted(p for p in KB_ROOT.iterdir() if p.is_dir())
    for agent_dir in agent_dirs:
        agent_name = agent_dir.name
        n = kb.load_agent(agent_name)
        print(f"  Loaded agent '{agent_name}': {n} chunk(s)")
    return kb


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Fern Health — Knowledge Base Loader")
    print("=" * 60)
    print(f"  KB root : {KB_ROOT}")
    print(f"  Chunk size : {CHUNK_SIZE} words  |  overlap : {CHUNK_OVERLAP} words")
    print()

    kb = get_kb()

    print()
    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    stats = kb.stats()
    if not stats:
        print("  (no documents loaded)")
    else:
        total_chunks = 0
        for agent, info in stats.items():
            print(f"  Agent : {agent}")
            print(f"    Documents : {info['documents']}")
            print(f"    Chunks    : {info['chunks']}")
            total_chunks += info["chunks"]
        print()
        print(f"  Total chunks in store : {total_chunks}")

    print()
    print("-" * 60)
    print("  Sample retrieval — query: 'treatment for painful periods'")
    print("-" * 60)
    results = kb.search("treatment for painful periods", top_k=2)
    if results:
        for rank, chunk in enumerate(results, 1):
            print(f"\n  [{rank}] {chunk.source}  (chunk {chunk.index})")
            preview = textwrap.fill(chunk.text[:300] + "…", width=56,
                                    initial_indent="      ",
                                    subsequent_indent="      ")
            print(preview)
    else:
        print("  (no results)")

    print()
    print("=" * 60)
    print("  KB ready — agents can call kb_loader.get_kb() to access it.")
    print("=" * 60)


if __name__ == "__main__":
    main()
