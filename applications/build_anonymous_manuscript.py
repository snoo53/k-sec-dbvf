"""Build the double-anonymized manuscript .docx for JMST submission.

Strips:
  - Author block (name, affiliation, ORCID, email)
  - CRediT statement that names the author
  - 'this lab', 'my prior', 'our prior' self-references that point to
    a previous publication of the author
"""

from __future__ import annotations

import re
from pathlib import Path

import pypandoc


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "JMST-DRAFT.md"
INTERMEDIATE = ROOT / "applications/_anon_manuscript.md"
OUT = ROOT / "applications/jmst_submission/Manuscript_anonymous.docx"


def anonymize(text: str) -> str:
    # Strip the author block (lines 3-5 in current draft)
    text = re.sub(
        r"^\*\*Sunwoo Lee\*\*¹\s*\n\s*\n¹[^\n]+\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Replace any remaining ORCID / email / name in body or sections
    text = re.sub(r"Sunwoo Lee", "[Author]", text)
    text = re.sub(r"0009-0004-9159-367X", "[ORCID redacted]", text)
    text = re.sub(r"lee\.11539@buckeyemail\.osu\.edu", "[email redacted]", text)

    # CRediT block — replace named author with neutral language
    text = re.sub(
        r"\*\*\[Author\]\*\*: ",
        "**The author**: ",
        text,
    )

    # Self-references to prior work
    # "my prior IonPath dual-graph" -> "a prior dual-graph baseline"
    text = re.sub(
        r"my prior IonPath dual-graph model",
        "a prior dual-graph baseline",
        text,
    )
    text = re.sub(
        r"the prior IonPath dual-graph model",
        "the prior dual-graph baseline",
        text,
    )
    # Table label "IonPath dual-graph (this lab, prior)" -> "Dual-graph (prior baseline)"
    text = re.sub(
        r"IonPath dual-graph \(this lab, prior\)",
        "Dual-graph baseline (prior)",
        text,
    )
    text = re.sub(
        r"IonPath dual-graph",
        "Dual-graph baseline",
        text,
    )
    # "this lab" anywhere
    text = re.sub(r"this lab", "the author's prior work", text)

    return text


def build():
    md = SRC.read_text(encoding="utf-8")
    anon = anonymize(md)
    INTERMEDIATE.write_text(anon, encoding="utf-8")
    pypandoc.convert_file(
        str(INTERMEDIATE), "docx",
        outputfile=str(OUT),
        extra_args=[
            "--standalone",
            "--from=markdown+pipe_tables+raw_html+tex_math_dollars",
            "--to=docx",
        ],
    )
    print(f"wrote {OUT}")
    INTERMEDIATE.unlink()


if __name__ == "__main__":
    build()
