"""
KBLoader and RAGRetriever for Agent 3 treatment knowledge base.

KBLoader: loads and validates JSON treatment entries from a directory.
RAGRetriever: keyword-scored retrieval over loaded entries with condition filtering.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


class KBLoader:
    """
    Loads treatment knowledge base from JSON files in a directory.
    Each file must contain a list of treatment objects or a single treatment object.
    """

    def __init__(self, kb_dir: str):
        self.kb_dir = Path(kb_dir)
        self._entries: list[dict] = []
        self._loaded = False
        self._load()

    def _load(self):
        """Load all JSON files from the KB directory."""
        self._entries = []
        if not self.kb_dir.exists():
            return

        for json_file in sorted(self.kb_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._entries.extend(data)
                elif isinstance(data, dict):
                    self._entries.append(data)
            except (json.JSONDecodeError, OSError):
                pass  # skip malformed files

        self._loaded = True

    def get_entries(self) -> list[dict]:
        """Return all loaded treatment entries."""
        return list(self._entries)

    def validate(self) -> dict:
        """
        Validate the KB and return summary info.
        Returns: {"entries": int, "conditions": list[str]}
        """
        conditions: set[str] = set()
        for entry in self._entries:
            for cond in entry.get("conditions", []):
                conditions.add(cond)
        return {
            "entries": len(self._entries),
            "conditions": sorted(conditions),
        }


class RAGRetriever:
    """
    Keyword-based retriever over KBLoader entries.
    Supports condition filtering and query-based scoring.
    """

    def __init__(self, kb_loader: KBLoader):
        self._loader = kb_loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        conditions: list[str] = None,
        current_treatments: list[str] = None,
        top_k: int = 15,
    ) -> dict:
        """
        Retrieve the most relevant treatments for a query.

        Args:
            query: Free-text semantic search query.
            conditions: Condition filter list (e.g. ["endometriosis"]).
            current_treatments: Currently used treatments (for contraindication lookup).
            top_k: Maximum number of treatments to return.

        Returns:
            {
                "retrieval_mode": str,
                "query": str,
                "conditions_filter": list,
                "retrieved_count": int,
                "total_kb_entries": int,
                "treatments": list[dict],
                "contraindications": list[dict],
            }
        """
        if conditions is None:
            conditions = []
        if current_treatments is None:
            current_treatments = []

        all_entries = self._loader.get_entries()
        total = len(all_entries)

        # Step 1 — condition filter
        if conditions:
            normalized = [c.lower().strip() for c in conditions]
            pool = [
                e for e in all_entries
                if any(c in [x.lower() for x in e.get("conditions", [])] for c in normalized)
            ]
            retrieval_mode = "condition_filtered"
        else:
            pool = list(all_entries)
            retrieval_mode = "full_scan"

        # Step 2 — score by keyword overlap with query
        scored = self._score(pool, query)

        # Step 3 — sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [entry for entry, _ in scored[:top_k]]

        # Step 4 — extract contraindications relevant to current treatments
        contraindications = self._find_contraindications(top, current_treatments)

        return {
            "retrieval_mode": retrieval_mode,
            "query": query,
            "conditions_filter": conditions,
            "retrieved_count": len(top),
            "total_kb_entries": total,
            "treatments": top,
            "contraindications": contraindications,
        }

    def build_prompt_context(self, retrieval_result: dict) -> str:
        """
        Convert a retrieval result into a formatted string for injection
        into the drafter system prompt.
        """
        treatments = retrieval_result["treatments"]
        contraindications = retrieval_result["contraindications"]

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("RETRIEVED KNOWLEDGE BASE — TREATMENT OPTIONS")
        lines.append(f"(Retrieved {retrieval_result['retrieved_count']} of "
                     f"{retrieval_result['total_kb_entries']} total entries)")
        lines.append("=" * 60)
        lines.append("")

        # Group by evidence tier
        tier_order = ["well_evidenced", "promising", "emerging"]
        tier_labels = {
            "well_evidenced": "WELL-EVIDENCED OPTIONS",
            "promising": "PROMISING OPTIONS",
            "emerging": "EMERGING OPTIONS",
        }

        by_tier: dict[str, list[dict]] = {t: [] for t in tier_order}
        for t in treatments:
            tier = t.get("evidence_tier", "emerging")
            if tier in by_tier:
                by_tier[tier].append(t)
            else:
                by_tier.setdefault(tier, []).append(t)

        for tier in tier_order:
            entries = by_tier[tier]
            if not entries:
                continue
            lines.append(f"### {tier_labels[tier]}")
            lines.append("")
            for entry in entries:
                lines.append(self._format_entry(entry))
                lines.append("")

        if contraindications:
            lines.append("### CONTRAINDICATION ALERTS")
            lines.append("")
            for ci in contraindications:
                lines.append(
                    f"- **{ci['treatment']}**: {ci['note']} "
                    f"(current treatment: {ci['current_treatment']})"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, entries: list[dict], query: str) -> list[tuple[dict, float]]:
        """Score each entry by keyword overlap with the query."""
        if not query.strip():
            return [(e, 1.0) for e in entries]

        keywords = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))

        scored = []
        for entry in entries:
            # Build a searchable blob from fields that matter
            blob_parts = [
                entry.get("name", ""),
                entry.get("description", ""),
                entry.get("mechanism", ""),
                " ".join(entry.get("tags", [])),
                " ".join(entry.get("conditions", [])),
            ]
            # Add nested text
            ovi = entry.get("ovarian_health_impact", {})
            if isinstance(ovi, dict):
                blob_parts.append(ovi.get("notes", ""))
            hc = entry.get("holistic_considerations", {})
            if isinstance(hc, dict):
                blob_parts.append(hc.get("notes", ""))

            blob = " ".join(blob_parts).lower()
            blob_words = set(re.findall(r"[a-zA-Z]{3,}", blob))

            overlap = len(keywords & blob_words)
            # Boost well-evidenced treatments slightly
            tier_boost = {"well_evidenced": 0.5, "promising": 0.25, "emerging": 0.0}.get(
                entry.get("evidence_tier", "emerging"), 0.0
            )
            score = float(overlap) + tier_boost
            scored.append((entry, score))

        return scored

    def _find_contraindications(
        self, treatments: list[dict], current_treatments: list[str]
    ) -> list[dict]:
        """
        Match current treatment names against each treatment's contraindications.
        Returns a list of alert dicts.
        """
        if not current_treatments:
            return []

        alerts = []
        current_lower = [t.lower() for t in current_treatments]

        for entry in treatments:
            for ci in entry.get("contraindications", []):
                ci_lower = ci.lower()
                for curr in current_lower:
                    # Simple substring match
                    if curr in ci_lower or ci_lower in curr:
                        alerts.append(
                            {
                                "treatment": entry["name"],
                                "current_treatment": curr,
                                "note": ci,
                            }
                        )
                        break  # one alert per treatment per current med is enough

        return alerts

    def _format_entry(self, entry: dict) -> str:
        """Format a single KB entry as a readable block."""
        lines = []
        name = entry.get("name", "Unknown")
        conditions = ", ".join(entry.get("conditions", []))
        tier = entry.get("evidence_tier", "unknown")
        lines.append(f"**{name}**")
        lines.append(f"  conditions: {conditions} | evidence_tier: {tier}")
        lines.append(f"  description: {entry.get('description', '')}")
        lines.append(f"  mechanism: {entry.get('mechanism', '')}")

        ovi = entry.get("ovarian_health_impact", {})
        if isinstance(ovi, dict) and ovi:
            sup = ovi.get("suppresses_ovulation", False)
            notes = ovi.get("notes", "")
            lines.append(
                f"  ovarian_health_impact: suppresses_ovulation={sup}; {notes}"
            )

        hc = entry.get("holistic_considerations", {})
        if isinstance(hc, dict) and hc:
            deps = hc.get("nutrient_depletions", [])
            notes = hc.get("notes", "")
            if deps:
                lines.append(
                    f"  holistic_considerations: nutrient_depletions={', '.join(deps)}; {notes}"
                )
            elif notes:
                lines.append(f"  holistic_considerations: {notes}")

        ci_list = entry.get("contraindications", [])
        if ci_list:
            lines.append(f"  contraindications: {'; '.join(ci_list)}")

        se = entry.get("side_effects", "")
        if se:
            lines.append(f"  side_effects: {se}")

        cost = entry.get("cost", "")
        if cost:
            lines.append(f"  cost: {cost}")

        return "\n".join(lines)
