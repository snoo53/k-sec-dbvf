"""WP1: Literature-mine SSE σ values using an LLM.

Given a list of DOIs (or text abstracts) of post-2023 SSE papers, query
an LLM (Claude/GPT-4) to extract (composition, temperature_C, σ_S_cm)
triples. Save to parquet for merging into the unified σ dataset.

INPUT: data/raw/literature_dois.txt (one DOI per line; user-supplied)
OUTPUT: data/raw/literature_mined.parquet

Manual-curation checklist for each extracted row:
  - composition parses with pymatgen.Composition
  - temperature is numeric Celsius
  - σ is positive, within [1e-15, 1e0] S/cm
  - source DOI is retained

Usage:
    ANTHROPIC_API_KEY=... python scripts/13_literature_mine.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are extracting ionic-conductivity data from a
scientific paper about solid-state electrolytes.

From the paper abstract + introduction + results I provide, extract every
(composition, temperature, ionic conductivity) triple you can find.

Rules:
1. `composition`: the full chemical formula as written (e.g. "Li6.4La3Zr1.4Ta0.6O12").
   Preserve non-integer stoichiometries. Do not normalize.
2. `temperature_C`: temperature in Celsius. If the paper reports in Kelvin, convert.
   If "room temperature" is written without a number, use 25.
3. `sigma_S_per_cm`: ionic conductivity in S/cm. If the paper reports S/m, divide by 100.
   If the paper reports as log10(σ), convert back to linear.
4. Only include values for the **bulk (total) ionic conductivity** of the solid electrolyte.
   Do NOT include partial conductivities, interface impedance, or activation energies.
5. If a value is approximate (e.g. "~10⁻³ S/cm"), extract it anyway but set `confidence=0.5`.
6. Ion chemistry: state which mobile ion (Li, Na, K, Mg, Ca). Skip if not alkali.

Output strict JSON: a list of objects with keys
{composition, temperature_C, sigma_S_per_cm, mobile_ion, confidence, context_quote}
If you find no valid triples, output an empty list: [].
"""


def query_claude(paper_text: str, api_key: str, model: str = "claude-sonnet-4-6") -> list[dict]:
    """Send one paper's text to Claude, parse extracted triples."""
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "user", "content": EXTRACTION_PROMPT + "\n\nPAPER TEXT:\n" + paper_text[:60000]}
        ],
    )
    # Claude sometimes wraps in ```json ... ``` — strip.
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    try:
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("parse fail on response: %s", exc)
        return []


def validate_triple(t: dict) -> dict | None:
    """Return normalized triple or None if invalid."""
    try:
        from pymatgen.core import Composition
    except ImportError:
        sys.exit("pip install pymatgen")
    try:
        comp = Composition(t["composition"]).reduced_formula
        T = float(t["temperature_C"])
        sigma = float(t["sigma_S_per_cm"])
    except Exception:
        return None
    if not (-20 < T < 900):
        return None
    if not (1e-18 < sigma < 1e1):
        return None
    mi = t.get("mobile_ion", "Li")
    if mi not in {"Li", "Na", "K", "Mg", "Ca"}:
        return None
    import math
    return dict(
        composition=t["composition"],
        reduced_formula=comp,
        temperature_C=T,
        sigma_S_per_cm=sigma,
        log_sigma=float(math.log10(sigma)),
        mobile_ion=mi,
        confidence=float(t.get("confidence", 1.0)),
        context_quote=t.get("context_quote", "")[:300],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/raw/papers",
                   help="Directory of .txt files, one per paper (user-supplied)")
    p.add_argument("--out", default="data/raw/literature_mined.parquet")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--dry-run", action="store_true",
                   help="List inputs without calling the LLM")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("Set ANTHROPIC_API_KEY in env")

    paper_dir = Path(args.input_dir)
    if not paper_dir.exists():
        log.warning("input dir %s does not exist — nothing to mine", paper_dir)
        log.info("To use: place plain-text paper files (abstract + intro + results) "
                 "in %s, one per paper.", paper_dir)
        return

    paper_files = sorted(paper_dir.glob("*.txt"))
    log.info("found %d paper files in %s", len(paper_files), paper_dir)

    if args.dry_run:
        for f in paper_files[:10]:
            log.info("  %s  (%.1fk chars)", f.name, len(f.read_text(encoding="utf-8")) / 1000)
        return

    rows = []
    for i, f in enumerate(paper_files):
        text = f.read_text(encoding="utf-8")
        try:
            triples = query_claude(text, api_key, args.model)
        except Exception as exc:  # noqa: BLE001
            log.error("  %s  LLM call failed: %s", f.name, exc)
            continue
        n_valid = 0
        for t in triples:
            norm = validate_triple(t)
            if norm:
                norm["source"] = f.stem
                rows.append(norm)
                n_valid += 1
        log.info("  [%d/%d] %s  extracted %d (kept %d)", i + 1, len(paper_files), f.stem, len(triples), n_valid)
        time.sleep(0.5)  # be polite to API

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out)
        log.info("saved %d rows to %s", len(rows), args.out)
    else:
        log.info("no valid rows extracted")


if __name__ == "__main__":
    main()
