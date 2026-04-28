"""Convert markdown-source-tree figure references to journal-style.

Body text: 'figs/fig_5_parity_per_bin.png' -> 'Figure 5'
Captions:  trailing 'See `figs/fig_N_*.png`.' is removed (file reference is
           uploaded separately to Editorial Manager; not part of the caption).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "JMST-DRAFT.md"


# Map fig_<N>_<rest>.png -> N. The 'fig_5_kubic_interpret' is special: it's
# Figure 12 in the manuscript (cubic-harmonic SI fig).
SI_FIG_NAME_TO_NUMBER = {
    "fig_5_kubic_interpret": 12,  # SI figure
}

REF_PATH_RE = re.compile(r"`?figs/(fig_(\d+)_[\w-]+)\.png`?")


def number_for(stem: str, num_str: str) -> int:
    if stem in SI_FIG_NAME_TO_NUMBER:
        return SI_FIG_NAME_TO_NUMBER[stem]
    return int(num_str)


def transform(text: str) -> str:
    # Step 1: drop trailing "See `figs/fig_N_*.png`." in figure-captions section.
    # Match the full sentence "See `figs/...png`." with optional whitespace.
    text = re.sub(
        r"\s*See\s*`?figs/fig_\d+_[\w-]+\.png`?\s*\.\s*",
        " ",
        text,
    )
    # Step 2: replace remaining inline path references with figure-number text.
    def sub(m):
        stem = m.group(1)
        n = number_for(stem, m.group(2))
        return f"Fig. {n}"
    text = REF_PATH_RE.sub(sub, text)
    # Tidy double spaces / spaces before punctuation introduced by deletions.
    text = re.sub(r"  +", " ", text)
    text = re.sub(r" +\n", "\n", text)
    return text


def main():
    orig = SRC.read_text(encoding="utf-8")
    new = transform(orig)
    if new == orig:
        print("no changes")
        return
    bak = SRC.with_suffix(SRC.suffix + ".bak2")
    bak.write_text(orig, encoding="utf-8")
    SRC.write_text(new, encoding="utf-8")
    # crude diff stat
    n_paths_before = len(REF_PATH_RE.findall(orig))
    n_paths_after = len(REF_PATH_RE.findall(new))
    print(f"replaced {n_paths_before - n_paths_after} path references "
          f"(remaining: {n_paths_after})")
    print(f"backup at {bak}")


if __name__ == "__main__":
    main()
