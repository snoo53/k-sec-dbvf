# JMST Submission — Portal Walk-Through

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
