"""JSONL persistence for search results."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ttc_operatorbench.core.schema import SearchResult


def write_search_results_jsonl(
    path: Path,
    results: Iterable[SearchResult],
    *,
    append: bool = False,
) -> Path:
    """Write search results to JSONL, one result per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as file:
        for result in results:
            file.write(result.model_dump_json())
            file.write("\n")
    return path


def read_search_results_jsonl(path: Path) -> tuple[SearchResult, ...]:
    """Read search results from JSONL."""
    results: list[SearchResult] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                results.append(SearchResult.model_validate_json(stripped))
    return tuple(results)


__all__ = ["read_search_results_jsonl", "write_search_results_jsonl"]
