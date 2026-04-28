"""Convert sole-author plural pronouns (we/our/us) to singular (I/my/me).

Used for: JMST-DRAFT.md, JMST-SI.md.
Handles edge cases: 'we are' → 'I am', 'we've' → 'I've', etc.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# Each replacement is applied with a word-boundary regex.
# Order matters: handle compounds (we've, we're) BEFORE bare 'we'.
REPLACEMENTS = [
    # Contractions
    (r"\bWe've\b", "I've"),
    (r"\bwe've\b", "I've"),
    (r"\bWe're\b", "I'm"),
    (r"\bwe're\b", "I'm"),
    (r"\bWe'll\b", "I'll"),
    (r"\bwe'll\b", "I'll"),
    (r"\bWe'd\b", "I'd"),
    (r"\bwe'd\b", "I'd"),
    # 'we are' → 'I am' (case-preserving, 2 forms)
    (r"\bWe are\b", "I am"),
    (r"\bwe are\b", "I am"),
    (r"\bWe were\b", "I was"),
    (r"\bwe were\b", "I was"),
    # bare pronouns
    (r"\bWe\b", "I"),
    (r"\bwe\b", "I"),
    (r"\bOur\b", "My"),
    (r"\bour\b", "my"),
    (r"\bOurs\b", "Mine"),
    (r"\bours\b", "mine"),
    (r"\bUs\b", "Me"),  # very rare in academic prose; leave word-boundary regex to handle
    (r"\bus\b", "me"),  # cautious — may need manual review for "let us"
    # collective phrases
    (r"\bthe authors declare\b", "the author declares"),
    (r"\bThe authors declare\b", "The author declares"),
    (r"\bAll authors\b", "I"),
    (r"\ball authors\b", "I"),
]


def transform(text: str) -> str:
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    return text


def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    new = transform(text)
    if new == text:
        print(f"  no changes: {path}")
        return
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    # crude diff stat
    n_changes = sum(text.count(p[0].replace(r"\b", "")) for p in REPLACEMENTS)
    print(f"  rewrote {path}")


def main():
    targets = [ROOT / "JMST-DRAFT.md", ROOT / "JMST-SI.md"]
    for p in targets:
        if p.exists():
            process_file(p)


if __name__ == "__main__":
    main()
