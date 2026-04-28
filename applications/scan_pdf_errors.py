"""Scan ChemRxiv-Manuscript.pdf for unrendered glyphs / LaTeX leakage."""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "ChemRxiv-Manuscript.pdf"

r = PdfReader(str(PDF))
text = ""
for i, page in enumerate(r.pages):
    text += f"\n--- page {i + 1} ---\n" + page.extract_text()

# Suspicious replacement glyphs
suspicious = {
    "U+FFFD (replacement char)": "�",
    "U+25A1 (square)": "□",
    "U+FFFC (object replacement)": "￼",
}
print("=== suspicious replacement glyphs ===")
for label, ch in suspicious.items():
    n = text.count(ch)
    print(f"  {'FOUND' if n else 'ok':<5} {label}: {n}")

# Math/Greek/super/sub characters that should have been replaced
print("\n=== residual unrendered math/Greek chars ===")
codepoints = [
    0x2264, 0x2265, 0x2272, 0x2273,
    0x03C3, 0x03C1, 0x0394, 0x03A3, 0x03C0, 0x03C4, 0x03A9,
    0x2208, 0x2282, 0x2295, 0x2192, 0x2190, 0x2297,
    0x2080, 0x2081, 0x2082, 0x2083, 0x2084,
    0x00C5, 0x00B2, 0x00B3,
]
any_residual = False
for cp in codepoints:
    n = text.count(chr(cp))
    if n:
        any_residual = True
        idx = text.find(chr(cp))
        ctx = text[max(0, idx - 40): idx + 40].replace("\n", " ")
        print(f"  U+{cp:04X} '{chr(cp)}' x{n}    ctx: ...{ctx}...")
if not any_residual:
    print("  ok: zero residual math/Greek chars")

# Unrendered LaTeX commands leaking through
print("\n=== LaTeX command leakage ===")
patterns = [
    (r"\\(?:sigma|rho|leq|geq|in|to|times|cdot|pm|mathbb|approx|hat|Delta|Sigma|tau|pi|otimes|subset|oplus|lesssim|gtrsim)",
     r"\command"),
    (r"\\AA", r"\AA literal"),
    (r"\$[^$\n]{0,20}\$", "math span (LaTeX-rendered would not show $..$)"),
]
for pat, label in patterns:
    matches = re.findall(pat, text)
    print(f"  {label}: {len(matches)}")
    if matches:
        for m in set(matches[:5]):
            idx = text.find(m)
            ctx = text[max(0, idx - 40): idx + 40].replace("\n", " ")
            print(f"    {m}: ...{ctx}...")

# Stray rendering issues
print("\n=== other patterns ===")
print(f"  '??' (consecutive): {text.count('??')}")
qmark_count = len(re.findall(r" \? ", text))
print(f"  ' ? ' isolated: {qmark_count}")
print(f"  empty math spans like '$$' : {text.count('$$')}")

# Total length sanity check
print(f"\n  total text characters extracted: {len(text):,}")
print(f"  pages: {len(r.pages)}")
