"""Build a single all-in-one PDF for ChemRxiv submission.

ChemRxiv accepts one main PDF that includes title page, full manuscript,
figures, references, and (optionally) SI. Author identity is visible.

Strategy:
  1. Compose a master markdown that:
       - Opens with the title + author block + abstract + keywords (visible)
       - Includes the full manuscript body
       - Embeds figures inline at the end of the body (one figure per page)
       - Appends the full SI as an "Appendix" section
  2. Convert with pandoc + xelatex.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pypandoc


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "applications/jmst_submission"
PKG_FIGS = PKG / "Figures"
OUT_PDF = ROOT / "ChemRxiv-Manuscript.pdf"
INTERMEDIATE = ROOT / "applications/_chemrxiv_master.md"


UNICODE_MAP = [
    # Greek letters (use chr() codepoints to bypass any source-encoding issues)
    (chr(0x03C3), r"$\sigma$"),     # σ
    (chr(0x03C1), r"$\rho$"),       # ρ
    (chr(0x03C0), r"$\pi$"),        # π
    (chr(0x03C4), r"$\tau$"),       # τ
    (chr(0x0394), r"$\Delta$"),     # Δ
    (chr(0x03A3), r"$\Sigma$"),     # Σ
    (chr(0x03A9), r"$\Omega$"),     # Ω
    (chr(0x03B5), r"$\varepsilon$"), # ε
    (chr(0x00F8), r"\o "),          # ø
    (chr(0x00D8), r"\O "),          # Ø
    # Subscripts U+2080..U+2089
    *[(chr(0x2080 + i), f"$_{i}$") for i in range(10)],
    (chr(0x2090), "$_a$"), (chr(0x1D66), "$_b$"),
    # Superscripts U+00B2..U+00B3, U+2070..U+2079
    (chr(0x00B2), "$^2$"), (chr(0x00B3), "$^3$"), (chr(0x00B9), "$^1$"),
    *[(chr(0x2070 + i), f"$^{i}$") for i in (0, 4, 5, 6, 7, 8, 9)],
    (chr(0x207B), "$^-$"), (chr(0x207A), "$^+$"),
    (chr(0x02E3), "$^x$"),
    # Math symbols
    (chr(0x2208), r"$\in$"),     # ∈
    (chr(0x2282), r"$\subset$"),  # ⊂
    (chr(0x2295), r"$\oplus$"),   # ⊕
    (chr(0x2265), r"$\geq$"),     # ≥
    (chr(0x2264), r"$\leq$"),     # ≤
    (chr(0x2272), r"$\lesssim$"), # ≲
    (chr(0x2273), r"$\gtrsim$"),  # ≳
    (chr(0x2248), r"$\approx$"),  # ≈
    (chr(0x2260), r"$\ne$"),      # ≠
    (chr(0x2192), r"$\to$"),      # →
    (chr(0x2190), r"$\leftarrow$"), # ←
    (chr(0x00D7), r"$\times$"),   # ×
    (chr(0x00B7), r"$\cdot$"),    # ·
    (chr(0x00B1), r"$\pm$"),      # ±
    (chr(0x2297), r"$\otimes$"),  # ⊗
    # Blackboard bold
    (chr(0x2102), r"$\mathbb{C}$"),  # ℂ
    (chr(0x211D), r"$\mathbb{R}$"),  # ℝ
    (chr(0x2124), r"$\mathbb{Z}$"),  # ℤ
    # Misc
    (chr(0x00C5), r"\AA{}"),      # Å
    # k̂ is "k" + combining circumflex U+0302
    ("k" + chr(0x0302), r"$\hat{k}$"),
]


# ASCII fallback for code blocks (where math mode doesn't apply)
ASCII_MAP = [
    (chr(0x03C3), "sigma"), (chr(0x03C1), "rho"), (chr(0x03C0), "pi"),
    (chr(0x03C4), "tau"), (chr(0x0394), "Delta"), (chr(0x03A3), "Sigma"),
    (chr(0x03A9), "Omega"), (chr(0x03B5), "epsilon"),
    (chr(0x00F8), "o"), (chr(0x00D8), "O"),
    *[(chr(0x2080 + i), str(i)) for i in range(10)],
    (chr(0x2090), "_a"), (chr(0x1D66), "_b"),
    (chr(0x00B2), "^2"), (chr(0x00B3), "^3"), (chr(0x00B9), "^1"),
    *[(chr(0x2070 + i), f"^{i}") for i in (0, 4, 5, 6, 7, 8, 9)],
    (chr(0x207B), "^-"), (chr(0x207A), "^+"), (chr(0x02E3), "^x"),
    (chr(0x2208), " in "), (chr(0x2282), " subset "),
    (chr(0x2295), " (+) "), (chr(0x2265), ">="),
    (chr(0x2264), "<="), (chr(0x2272), "<~"), (chr(0x2273), ">~"),
    (chr(0x2248), "~~"), (chr(0x2260), "!="),
    (chr(0x2192), "->"), (chr(0x2190), "<-"),
    (chr(0x00D7), "x"), (chr(0x00B7), "*"), (chr(0x00B1), "+/-"),
    (chr(0x2297), "(x)"),
    (chr(0x2102), "C"), (chr(0x211D), "R"), (chr(0x2124), "Z"),
    (chr(0x00C5), "A"),
    ("k" + chr(0x0302), "k_hat"),
]


def _apply_map(text: str, mapping) -> str:
    for u, repl in mapping:
        text = text.replace(u, repl)
    return text


def replace_unicode(text: str) -> str:
    """Replace Unicode math/Greek chars with LaTeX equivalents in body text;
    use plain ASCII inside code regions (where math mode does not render).

    Code regions: triple-backtick fenced blocks AND single-backtick inline spans.
    """
    out_chunks = []

    # First, split off triple-backtick blocks line by line.
    in_fence = False
    line_bucket: list[tuple[bool, list[str]]] = []  # (in_fence?, lines)
    cur = (False, [])
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            line_bucket.append(cur)
            cur = (not in_fence, [line])  # the fence-marker line itself
            line_bucket.append(cur)
            cur = (not in_fence, [])
            in_fence = not in_fence
        else:
            cur[1].append(line)
    line_bucket.append(cur)

    # Process each region. Fenced => ASCII map straight. Non-fenced => walk
    # backtick-spans and apply ASCII map inside spans, UNICODE map outside.
    def process_inline(s: str) -> str:
        """Walk single-backtick spans within a non-fenced text chunk."""
        out = []
        i = 0
        while i < len(s):
            if s[i] == "`":
                # find matching closing backtick
                j = s.find("`", i + 1)
                if j == -1:
                    # no closer; treat rest as plain
                    out.append(_apply_map(s[i:], UNICODE_MAP))
                    break
                # span: s[i..j+1] inclusive of backticks
                span = s[i + 1: j]
                out.append("`" + _apply_map(span, ASCII_MAP) + "`")
                i = j + 1
            else:
                # advance until next backtick
                k = s.find("`", i)
                if k == -1:
                    out.append(_apply_map(s[i:], UNICODE_MAP))
                    break
                out.append(_apply_map(s[i:k], UNICODE_MAP))
                i = k
        return "".join(out)

    for is_fence, lines in line_bucket:
        if not lines:
            continue
        joined = "\n".join(lines)
        if is_fence:
            # whole region is inside a triple-backtick fence (or is the fence
            # marker line itself); use ASCII map
            out_chunks.append(_apply_map(joined, ASCII_MAP))
        else:
            out_chunks.append(process_inline(joined))

    return "\n".join(out_chunks)


def collect_master_markdown() -> str:
    """Compose the master markdown."""
    parts = []

    # Header / title page (author-visible)
    parts.append(
        "---\n"
        "title: \"Cubic-Equivariant k-Space Convolution and a Differentiable "
        "Bond-Valence Field for Ionic-Conductivity Prediction in "
        "Solid-State Electrolytes\"\n"
        "author:\n"
        "  - Sunwoo Lee\n"
        "date: April 2026\n"
        "geometry: margin=1in\n"
        "fontsize: 11pt\n"
        "linkcolor: black\n"
        "urlcolor: black\n"
        "citecolor: black\n"
        "---\n\n"
    )

    # Author block (since pandoc's frontmatter author is sparse)
    parts.append(
        "**Sunwoo Lee**¹\n\n"
        "¹ Independent researcher, South Korea. "
        "ORCID: 0009-0004-9159-367X. "
        "Correspondence: lee.11539@buckeyemail.osu.edu\n\n"
        "*Submitted to* Journal of Materials Science & Technology *(JMST). "
        "Preprint posted to ChemRxiv.*\n\n"
        "\\hrule\n\n"
    )

    # Manuscript body — the WHOLE JMST-DRAFT.md (it has author block too,
    # but the LaTeX template will render the title from the YAML frontmatter
    # and the inline title from the markdown will become a Heading 1; we
    # strip the duplicate title and author block.)
    main_md = (ROOT / "JMST-DRAFT.md").read_text(encoding="utf-8")

    # Drop the # title (1st heading) and the author/affiliation block under it
    # The author block is ~3 lines after the # title and ends before "## Highlights".
    lines = main_md.split("\n")
    out_lines = []
    skip = True
    for line in lines:
        if skip:
            if line.startswith("## Highlights"):
                skip = False
                out_lines.append(line)
                continue
            continue
        out_lines.append(line)
    body = "\n".join(out_lines)

    # Inline embed each figure right after its caption.
    figure_files = {
        1: "Fig_1_architecture.png",
        2: "Fig_2_baselines.png",
        3: "Fig_3_ablation.png",
        4: "Fig_4_dbvf_test.png",
        5: "Fig_5_parity_per_bin.png",
        6: "Fig_6_virtual_screen.png",
        7: "Fig_7_ood_by_family.png",
        8: "Fig_8_dbvf_learned.png",
        9: "Fig_9_learning_curve.png",
    }
    # Insert image embeds before each "**Figure N.**" caption marker.
    # Each becomes:  ![](path)\n\n**Figure N.** ...
    for n, fname in figure_files.items():
        path = PKG_FIGS / fname
        if not path.exists():
            print(f"WARNING: figure missing: {path}")
            continue
        path_posix = str(path).replace("\\", "/")
        marker = f"**Figure {n}.**"
        # Pandoc image syntax: ![alt](path){ attrs } — no quotes around path
        replacement = (
            f"\n\n![]({path_posix}){{ width=85% }}\n\n{marker}"
        )
        body = body.replace(marker, replacement, 1)

    parts.append(body)

    # Appendix: SI document
    parts.append("\n\n\\newpage\n\n# Appendix — Supplementary Information\n\n")
    si_md = (ROOT / "JMST-SI.md").read_text(encoding="utf-8")
    # Drop SI's own top-level title heading (we've already added an Appendix heading)
    si_lines = si_md.split("\n")
    si_clean = []
    for i, line in enumerate(si_lines):
        if line.startswith("# Supplementary Information"):
            continue
        si_clean.append(line)
    parts.append("\n".join(si_clean))

    # SI figures embedded at the end
    si_figs = {
        "S1": ("Fig_S1_calibration.png", "MC-dropout calibration"),
        "S2": ("Fig_S2_top_k_precision.png", "Top-K precision over OOF predictions"),
        "S3": ("Fig_S3_kubic_interpret.png", "Cubic-harmonic filter interpretability"),
    }
    parts.append("\n\n## SI figure plates\n\n")
    for tag, (fname, caption) in si_figs.items():
        path = PKG_FIGS / fname
        if path.exists():
            path_posix = str(path).replace("\\", "/")
            parts.append(
                f"![]({path_posix}){{ width=85% }}\n\n"
                f"**Figure {tag}.** *{caption}.*\n\n\\newpage\n\n"
            )

    return "".join(parts)


def main():
    md = collect_master_markdown()
    md = replace_unicode(md)
    # Pandoc treats $math$ NOT as math when the closing $ touches a digit
    # ($x$5 is parsed as currency, not math). Insert a thin space.
    import re as _re
    md = _re.sub(r"(\$[^$\n]+?\$)(\d)", r"\1 \2", md)
    INTERMEDIATE.write_text(md, encoding="utf-8")
    print(f"wrote master markdown: {INTERMEDIATE}")

    pypandoc.convert_file(
        str(INTERMEDIATE), "pdf",
        outputfile=str(OUT_PDF),
        extra_args=[
            "--pdf-engine=xelatex",
            "--from=markdown+pipe_tables+raw_html+tex_math_dollars+yaml_metadata_block",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "linkcolor=black",
            "-V", "urlcolor=black",
            "-V", "citecolor=black",
            "--standalone",
        ],
    )
    print(f"wrote {OUT_PDF}")
    INTERMEDIATE.unlink()


if __name__ == "__main__":
    main()
