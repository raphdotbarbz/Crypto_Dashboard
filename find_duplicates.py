#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import difflib
import hashlib
import os
import re
from pathlib import Path
from typing import List, Dict

# ---- Config ----
ROOT = Path(__file__).resolve().parents[0]
# Scan src/ by default; fall back to repo root if src/ doesn't exist.
INCLUDE_DIRS: List[Path] = [p for p in [ROOT / "src"] if p.exists()] or [ROOT]
EXCLUDE_PATTERNS = re.compile(r"/(\.venv|\.git|data|notebooks|reports)/")

# ---- AST visitor to collect functions and a normalized representation ----
class FuncUnit(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.funcs: list[tuple[str, int, str]] = []  # (name, lineno, normalized_src)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.funcs.append((node.name, node.lineno, self._normalize(node)))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.funcs.append((node.name, node.lineno, self._normalize(node)))
        self.generic_visit(node)

    def _normalize(self, node: ast.AST) -> str:
        """Normalize function source to compare structure across files."""
        src = ast.get_source_segment(self._raw_code, node) or ""
        # Drop docstrings and compress whitespace
        src = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", src, flags=re.S)
        src = re.sub(r"\s+", " ", src).strip()
        # Normalize throwaway identifiers (e.g., _tmp123 -> _id)
        src = re.sub(r"\b_[A-Za-z0-9]+\b", "_id", src)
        return src

    @property
    def _raw_code(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="ignore")


# ---- File walking ----
def walk_py_files() -> list[Path]:
    files: list[Path] = []
    for base in INCLUDE_DIRS:
        for root, _, fnames in os.walk(base):
            if EXCLUDE_PATTERNS.search(root.replace("\\", "/")):
                continue
            for f in fnames:
                if f.endswith(".py"):
                    files.append(Path(root) / f)
    return files


# ---- Collect normalized function units across the codebase ----
def collect_funcs():
    out: list[dict] = []
    for fpath in walk_py_files():
        try:
            tree = ast.parse(fpath.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        v = FuncUnit(fpath)
        v.visit(tree)
        for name, lineno, norm in v.funcs:
            h = hashlib.sha1(norm.encode()).hexdigest()
            out.append(
                {
                    "file": str(fpath),
                    "name": name,
                    "lineno": lineno,
                    "norm": norm,
                    "hash": h,
                }
            )
    return out


# ---- Main: exact and near-duplicate detection, CSV reports ----
def main():
    items = collect_funcs()

    # exact duplicates by hash
    hash_map: Dict[str, list[dict]] = {}
    for it in items:
        hash_map.setdefault(it["hash"], []).append(it)

    exact_dups = [v for v in hash_map.values() if len(v) > 1]

    # near duplicates by similarity threshold
    NEAR_T = 0.90
    near_dups: list[tuple[float, dict, dict]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["hash"] == b["hash"]:
                continue
            # Optional: prioritize common helper names (no-op placeholder)
            if a["name"] == b["name"] and a["name"] in {
                "get_prices",
                "calc_weights",
                "portfolio_returns",
            }:
                pass
            sim = difflib.SequenceMatcher(None, a["norm"], b["norm"]).ratio()
            if sim >= NEAR_T:
                near_dups.append((sim, a, b))

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    # CSV for exact
    with open(reports / "duplicates_exact.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "name", "lineno", "hash"])
        for group in exact_dups:
            for it in group:
                w.writerow([it["file"], it["name"], it["lineno"], it["hash"]])
            w.writerow([])

    # CSV for near
    with open(reports / "duplicates_near.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["similarity", "file_a", "name_a", "line_a", "file_b", "name_b", "line_b"]
        )
        for sim, a, b in sorted(near_dups, key=lambda x: -x[0]):
            w.writerow(
                [
                    f"{sim:.3f}",
                    a["file"],
                    a["name"],
                    a["lineno"],
                    b["file"],
                    b["name"],
                    b["lineno"],
                ]
            )

    print("Wrote reports/duplicates_exact.csv and reports/duplicates_near.csv")


if __name__ == "__main__":
    main()
