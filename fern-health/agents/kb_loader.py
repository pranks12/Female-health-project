import json
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "kb" / "agent3_treatment"


def load_kb() -> str:
    """Load all JSON files from the knowledge base directory and return as formatted text."""
    if not KB_DIR.exists():
        return ""

    entries = []
    for kb_file in sorted(KB_DIR.glob("*.json")):
        try:
            with open(kb_file) as f:
                data = json.load(f)
            entries.append(json.dumps(data, indent=2))
        except (json.JSONDecodeError, IOError):
            continue

    return "\n\n".join(entries) if entries else ""
