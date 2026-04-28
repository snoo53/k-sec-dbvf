"""Build the JMST submission package from the markdown sources.

Outputs into applications/jmst_submission/ :
    Manuscript.docx
    SupplementaryMaterial.docx
    Highlights.docx
    CoverLetter.pdf
    GraphicalAbstract.png       (= figs/fig_1_architecture.png)
    Figures/fig_*.png           (300-dpi PNG copies)
    refs.bib                    (BibTeX)
    SubmissionInstructions.md   (step-by-step Elsevier portal walk-through)

Then bundles everything into JMST-Submission-Package.zip at repo root.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pypandoc


ROOT = Path(__file__).resolve().parents[1]
APPS = ROOT / "applications"
PKG = APPS / "jmst_submission"
FIGS_SRC = ROOT / "figs"
PKG_FIGS = PKG / "Figures"


def clean():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True, exist_ok=True)
    PKG_FIGS.mkdir(parents=True, exist_ok=True)


def md_to_docx(md_path: Path, out_path: Path, extra_args=None):
    """Convert a markdown file to docx using pypandoc."""
    args = [
        "--standalone",
        "--from=markdown+pipe_tables+raw_html+tex_math_dollars",
        "--to=docx",
        "--toc-depth=2",
    ]
    if extra_args:
        args.extend(extra_args)
    pypandoc.convert_file(
        str(md_path), "docx",
        outputfile=str(out_path),
        extra_args=args,
    )
    print(f"wrote {out_path}")


def md_to_pdf(md_path: Path, out_path: Path):
    """Convert cover-letter markdown to PDF using reportlab."""
    # Import lazily so this module still loads if reportlab is missing.
    sys_path_inserted = False
    try:
        import sys
        if str(APPS) not in sys.path:
            sys.path.insert(0, str(APPS))
            sys_path_inserted = True
        import build_cover_pdf
        # build_cover_pdf writes to a fixed path; override SRC/OUT then call.
        build_cover_pdf.SRC = md_path
        build_cover_pdf.OUT = out_path
        build_cover_pdf.build()
    finally:
        if sys_path_inserted:
            sys.path.remove(str(APPS))


def write_highlights():
    text = (
        "# Highlights\n\n"
        "- k-SEC: cubic-equivariant reciprocal-space encoder for crystal "
        "property regression.\n"
        "- DBVF: end-to-end-learnable bond-valence module embedded in a "
        "neural network.\n"
        "- OBELiX neural-method MAE: 1.047 standalone, 0.980 stacked "
        "(n = 281, 5-fold CV).\n"
        "- DBVF features alone do not help LightGBM — the gain is "
        "architectural, not tabular.\n"
        "- Unsupervised top-15 screen of 18,574 MP crystals recovers "
        "known fast-Li families.\n"
    )
    src = APPS / "_highlights.md"
    src.write_text(text, encoding="utf-8")
    md_to_docx(src, PKG / "Highlights.docx")
    src.unlink()


def write_submission_instructions():
    instructions = """# JMST Submission — Portal Walk-Through

Submitting *Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence Field for Ionic-Conductivity Prediction in Solid-State Electrolytes* to *Journal of Materials Science & Technology* (Elsevier).

## Account setup (5 minutes)

1. Go to **https://www.editorialmanager.com/jmst/** (Elsevier Editorial Manager for JMST).
2. Click "Register Now" if you don't have an Elsevier account; otherwise "Login."
3. Use the email **lee.11539@buckeyemail.osu.edu** as your account email so the corresponding-author email matches the manuscript.
4. After registering, log in and choose role "Author."

## Start a new submission

5. Click "Submit New Manuscript."
6. Select Article Type: **Research Paper** (or "Original Research Article" if that's the listed type).

## Upload files (in this order — required by the portal)

For each file, the portal asks for an "Item" type. Match as below:

| Item type | Upload | Source file in this package |
|---|---|---|
| Manuscript Title Page | `TitlePage.docx` | `applications/jmst_submission/TitlePage.docx` |
| Cover Letter | `CoverLetter.pdf` (or `.docx`) | `applications/jmst_submission/CoverLetter.pdf` |
| Highlights | `Highlights.docx` | `applications/jmst_submission/Highlights.docx` |
| Manuscript | `Manuscript.docx` | `applications/jmst_submission/Manuscript.docx` |
| Graphical Abstract | `GraphicalAbstract.png` | `applications/jmst_submission/GraphicalAbstract.png` |
| Figure 1 — Architecture | `Fig_1_architecture.png` | `Figures/` |
| Figure 2 — Headline benchmark | `Fig_2_baselines.png` | `Figures/` |
| Figure 3 — Architectural ablation | `Fig_3_ablation.png` | `Figures/` |
| Figure 4 — DBVF as architecture | `Fig_4_dbvf_test.png` | `Figures/` |
| Figure 5 — Parity + per-σ-bin | `Fig_5_parity_per_bin.png` | `Figures/` |
| Figure 6 — Virtual screen | `Fig_6_virtual_screen.png` | `Figures/` |
| Figure 7 — OOD by family | `Fig_7_ood_by_family.png` | `Figures/` |
| Figure 8 — Learned bond-valence params | `Fig_8_dbvf_learned.png` | `Figures/` |
| Figure 9 — Learning curve | `Fig_9_learning_curve.png` | `Figures/` |
| Supplementary Material | `SupplementaryMaterial.docx` | `applications/jmst_submission/SupplementaryMaterial.docx` |
| SI Figure S1 — MC-dropout calibration | `Fig_S1_calibration.png` | `Figures/` |
| SI Figure S2 — Top-K precision | `Fig_S2_top_k_precision.png` | `Figures/` |
| SI Figure S3 — Cubic-harmonic filter | `Fig_S3_kubic_interpret.png` | `Figures/` |

## Manuscript metadata fields

- **Title.** Cubic-Equivariant k-Space Convolution and a Differentiable Bond-Valence Field for Ionic-Conductivity Prediction in Solid-State Electrolytes
- **Author 1.** Sunwoo Lee — Independent researcher, South Korea — ORCID 0009-0004-9159-367X — corresponding author — lee.11539@buckeyemail.osu.edu
- **Keywords (5–6).** solid electrolytes; lithium-ion conductivity; machine learning; bond-valence model; equivariant neural networks; OBELiX
- **Abstract.** Paste the manuscript abstract (200 words) into the abstract field.
- **Highlights.** Paste the 5 bullets into the highlights field; the file Highlights.docx is the formatted version.
- **Funding.** "This research received no external funding."
- **Conflict of interest.** "The author declares no competing interests."
- **AI-tools disclosure.** Copy the AI-tools paragraph from the manuscript verbatim into the disclosure field if there is one; if not, the manuscript's section already covers it.

## Suggested reviewers

Per JMST policy, suggest at least 4. Use the names from the cover letter:
1. An author of MACE / NequIP — e.g., Ilyes Batatia, Simon Batzner, or Gábor Csányi.
2. An author of CGCNN / ALIGNN / M3GNet — e.g., Tian Xie, Kamal Choudhary, Chi Chen.
3. A solid-state Li-electrolyte researcher — e.g., Yifei Mo, Shyue Ping Ong, Gerbrand Ceder.
4. A bond-valence-theory researcher — e.g., Stefan Adams (NUS).

You can leave email addresses blank; the editor can look them up. Or look up their group webpage and copy the public lab email.

## Final checks before clicking Submit

- [ ] Cover letter uploaded
- [ ] Manuscript uploaded with author block + ORCID + corresponding email
- [ ] Highlights uploaded (≤ 5 bullets, each ≤ 85 chars)
- [ ] Graphical abstract uploaded (Fig. 1)
- [ ] All 11 figures uploaded individually
- [ ] Supplementary material uploaded
- [ ] Suggested reviewers (4+) added
- [ ] Funding statement, COI, AI-tools disclosure filled in
- [ ] Abstract in metadata field matches manuscript
- [ ] Title in metadata field exactly matches manuscript
- [ ] Keywords in metadata field
- [ ] PDF preview reviewed (Editorial Manager builds a PDF from your uploads — read it before approving)

## After submitting

1. Note your **Manuscript ID** (e.g., JMST-D-26-XXXXX). Save it.
2. Submit to **arXiv** the same day for parallel preprint deposit:
   - Go to https://arxiv.org/submit
   - Category: cond-mat.mtrl-sci (primary), cs.LG (cross-list)
   - Upload the manuscript PDF + figures
3. Push the GitHub release with code, weights, and a README pointing at the arXiv ID.

Editorial response time at JMST is 2–4 weeks for the editorial decision (desk-reject or send to review). Peer review typically takes 2–4 months. You can check status anytime in Editorial Manager.

## If desk-rejected

Common reasons: out of scope, lack of novelty, English-language quality. Each is addressable. Possible alternative venues if JMST desk-rejects:
- *npj Computational Materials* (open access, IF ~10)
- *Chemistry of Materials* (ACS, IF ~7)
- *Journal of Materials Chemistry A* (RSC, IF ~11)
- *Digital Discovery* (RSC, IF ~6, open access)
- *Computational Materials Science* (Elsevier, IF ~3)
"""
    (PKG / "SubmissionInstructions.md").write_text(instructions, encoding="utf-8")
    print(f"wrote {PKG / 'SubmissionInstructions.md'}")


def copy_figures():
    """Copy figures and rename to match the FINAL figure numbering in the
    manuscript. The source-tree filenames reflect creation order, not the
    final layout, so we rename on copy to avoid uploader confusion."""
    rename_map = {
        # source-tree filename       ->  manuscript figure filename
        "fig_1_architecture.png":      "Fig_1_architecture.png",
        "fig_2_baselines.png":         "Fig_2_baselines.png",
        "fig_3_ablation.png":          "Fig_3_ablation.png",
        "fig_4_dbvf_test.png":         "Fig_4_dbvf_test.png",
        "fig_5_parity_per_bin.png":    "Fig_5_parity_per_bin.png",
        "fig_6_virtual_screen.png":    "Fig_6_virtual_screen.png",
        "fig_7_ood_by_family.png":     "Fig_7_ood_by_family.png",
        "fig_10_dbvf_learned.png":     "Fig_8_dbvf_learned.png",     # was 10
        "fig_11_learning_curve.png":   "Fig_9_learning_curve.png",   # was 11
        # SI figures
        "fig_8_calibration.png":       "Fig_S1_calibration.png",     # was 8
        "fig_9_top_k_precision.png":   "Fig_S2_top_k_precision.png", # was 9
        "fig_5_kubic_interpret.png":   "Fig_S3_kubic_interpret.png", # was 5 (collision)
    }
    for src_name, dst_name in rename_map.items():
        src = FIGS_SRC / src_name
        if src.exists():
            shutil.copy2(src, PKG_FIGS / dst_name)
        else:
            print(f"WARNING: missing {src}")
    shutil.copy2(FIGS_SRC / "fig_1_architecture.png", PKG / "GraphicalAbstract.png")
    print(f"copied {len(rename_map)} figures (renamed to match manuscript numbering) "
          f"+ GraphicalAbstract.png")


def copy_bibtex():
    shutil.copy2(ROOT / "refs.bib", PKG / "refs.bib")
    print("copied refs.bib")


def build_zip():
    """Create JMST-Submission-Package.zip at repo root."""
    out_zip = ROOT / "JMST-Submission-Package.zip"
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in PKG.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(PKG.parent)
                zf.write(path, arcname=str(arcname))
    size_mb = out_zip.stat().st_size / 1e6
    print(f"\nwrote {out_zip}  ({size_mb:.1f} MB)")


def main():
    clean()
    print("--- converting SI ---")
    md_to_docx(ROOT / "JMST-SI.md", PKG / "SupplementaryMaterial.docx")
    print("--- converting cover letter ---")
    md_to_docx(ROOT / "JMST-COVER-LETTER.md", PKG / "CoverLetter.docx")
    md_to_pdf(ROOT / "JMST-COVER-LETTER.md", PKG / "CoverLetter.pdf")
    print("--- highlights ---")
    write_highlights()
    print("--- title page ---")
    import sys
    if str(APPS) not in sys.path:
        sys.path.insert(0, str(APPS))
    import build_title_page
    build_title_page.OUT = PKG / "TitlePage.docx"
    build_title_page.build()
    print("--- anonymized manuscript (uploaded as 'Manuscript without author details') ---")
    import build_anonymous_manuscript
    build_anonymous_manuscript.OUT = PKG / "Manuscript.docx"
    build_anonymous_manuscript.build()
    print("--- declaration of interests (official Elsevier template) ---")
    target = PKG / "DeclarationOfInterests.docx"
    rc = subprocess.run([
        "curl", "-sSL", "-o", str(target),
        "https://legacyfileshare.elsevier.com/promis_misc/declaration-of-competing-interests.docx",
    ]).returncode
    if rc != 0 or not target.exists() or target.stat().st_size < 5000:
        raise RuntimeError("Failed to download Elsevier DOCI template")
    print(f"  downloaded official template to {target}")
    import fill_doci_template
    fill_doci_template.TARGET = target
    fill_doci_template.main()
    print("--- forcing black text on all .docx files ---")
    import build_anonymous_manuscript  # already imported above
    import force_black_text
    for docx in PKG.glob("*.docx"):
        force_black_text.process(docx)
    print("--- copying figures ---")
    copy_figures()
    print("--- copying refs.bib ---")
    copy_bibtex()
    print("--- writing submission instructions ---")
    write_submission_instructions()
    build_zip()


if __name__ == "__main__":
    main()
