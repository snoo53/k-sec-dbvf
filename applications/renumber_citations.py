"""Convert author-year citations to Vancouver numeric.

Handles:
  [Xie 2018]                                  -> [1]
  [Grinsztajn 2022; McElfresh 2023]           -> [11,12]
  [Adams 2022, Filsø 2013]                    -> [9,14]
  [Author Year]   wrapped across a newline    -> [n]

Usage:
    python applications/renumber_citations.py        # writes the file (with .bak)
    python applications/renumber_citations.py --dry  # preview the mapping only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "JMST-DRAFT.md"


REF_TEXT = {
    "Adams 2002": (
        "**Adams, S.** (2002). Bond valence analysis of structural "
        "preferences. *Solid State Ionics* 154-155, 151-159."
    ),
    "Adams 2022": (
        "**Adams, S.** (2022). Bond valence pathway analysis for "
        "ionic conductors. *Acta Crystallographica B* 78, 16-30."
    ),
    "Batatia 2022": (
        "**Batatia, I.; Kovács, D. P.; Simm, G. N. C.; Ortner, C.; "
        "Csányi, G.** (2022). MACE: Higher order equivariant message "
        "passing neural networks for fast and accurate force fields. "
        "*Adv. Neural Inf. Process. Syst.* 35."
    ),
    "Batatia 2024": (
        "**Batatia, I. *et al.*** (2024). A foundation model for "
        "atomistic materials chemistry (MACE-MP-0). *arXiv:2401.00096*."
    ),
    "Batzner 2022": (
        "**Batzner, S.; Musaelian, A.; Sun, L.; Geiger, M.; "
        "Mailoa, J. P.; Kornbluth, M.; Molinari, N.; Smidt, T. E.; "
        "Kozinsky, B.** (2022). E(3)-equivariant graph neural "
        "networks for data-efficient and accurate interatomic "
        "potentials (NequIP). *Nat. Commun.* 13, 2453."
    ),
    "Brown 2002": (
        "**Brown, I. D.** (2002). *The Chemical Bond in Inorganic "
        "Chemistry: The Bond Valence Model*. IUCr Monographs on "
        "Crystallography 12, Oxford University Press."
    ),
    "Chen 2019": (
        "**Chen, C.; Ye, W.; Zuo, Y.; Zheng, C.; Ong, S. P.** (2019). "
        "Graph networks as a universal machine learning framework "
        "for molecules and crystals (MEGNet). *Chem. Mater.* 31, "
        "3564-3572."
    ),
    "Chen 2022": (
        "**Chen, C.; Ong, S. P.** (2022). A universal graph deep "
        "learning interatomic potential for the periodic table "
        "(M3GNet). *Nat. Comput. Sci.* 2, 718-728."
    ),
    "Choudhary 2021": (
        "**Choudhary, K.; DeCost, B.** (2021). Atomistic line graph "
        "neural network for improved materials property predictions "
        "(ALIGNN). *npj Comput. Mater.* 7, 185."
    ),
    "Deng 2023": (
        "**Deng, B.; Zhong, P.; Jun, K.; Riebesell, J.; Han, K.; "
        "Bartel, C. J.; Ceder, G.** (2023). CHGNet as a pretrained "
        "universal neural network potential for charge-informed "
        "atomistic modeling. *Nat. Mach. Intell.* 5, 1031-1041."
    ),
    "Filsø 2013": (
        "**Filsø, M. Ø.; Turner, M. J.; Gibbs, G. V.; Adams, S.; "
        "Spackman, M. A.; Iversen, B. B.** (2013). Visualizing "
        "lithium-ion migration pathways by bond-valence-energy "
        "landscapes. *Chem. Eur. J.* 19, 15535-15544."
    ),
    "Grinsztajn 2022": (
        "**Grinsztajn, L.; Oyallon, E.; Varoquaux, G.** (2022). Why "
        "do tree-based models still outperform deep learning on "
        "tabular data? *Adv. Neural Inf. Process. Syst.* 35."
    ),
    "Hargreaves 2023": (
        "**Hargreaves, J. *et al.*** (2023). A database of "
        "experimentally measured lithium solid electrolyte "
        "conductivities. *Sci. Data* 10, 471."
    ),
    "Hollmann 2025": (
        "**Hollmann, N.; Müller, S.; Eggensperger, K.; Hutter, F.** "
        "(2025). Accurate predictions on small data with a tabular "
        "foundation model (TabPFN). *Nature*."
    ),
    "Li 2020": (
        "**Li, Z.; Kovachki, N.; Azizzadenesheli, K.; Liu, B.; "
        "Bhattacharya, K.; Stuart, A.; Anandkumar, A.** (2020). "
        "Fourier neural operator for parametric partial "
        "differential equations. *arXiv:2010.08895*."
    ),
    "McElfresh 2023": (
        "**McElfresh, D.; Khandagale, S.; Valverde, J.; "
        "Prasad C, V.; Ramakrishnan, G.; Goldblum, M.; White, C.** "
        "(2023). When do neural nets outperform boosted trees on "
        "tabular data? *Adv. Neural Inf. Process. Syst.* 36."
    ),
    "Pizarro 2025": (
        "**Pizarro, F. *et al.*** (2025). OBELiX: a curated "
        "benchmark for crystal-structured Li-ion solid electrolytes. "
        "*arXiv:2502.14234*."
    ),
    "Wang 2021": (
        "**Wang, A. Y.-T.; Kauwe, S. K.; Murdock, R. J.; "
        "Sparks, T. D.** (2021). Compositionally restricted "
        "attention-based network for materials property prediction "
        "(CrabNet). *npj Comput. Mater.* 7, 77."
    ),
    "Xie 2018": (
        "**Xie, T.; Grossman, J. C.** (2018). Crystal graph "
        "convolutional neural networks for accurate and "
        "interpretable prediction of material properties (CGCNN). "
        "*Phys. Rev. Lett.* 120, 145301."
    ),
    "Yan 2025": (
        "**Yan, K. *et al.*** (2025). ReGNet: Reciprocal-space "
        "neural networks for crystal property prediction. "
        "*arXiv:2502.02748*."
    ),
}

ALIAS = {
    "ChemArr 2023": "Hargreaves 2023",
}


# Match a [...] group that contains at least one author-year pattern.
# Inside, names may include unicode chars (Filsø) and may wrap on a newline.
GROUP_RE = re.compile(r"\[([^\[\]]*?\d{4}[^\[\]]*?)\]")
# Inside a group, split on , or ; and trim. Each entry is "Author Year".
ENTRY_RE = re.compile(r"^([\wÀ-ſ][\wÀ-ſ .'-]+?)\s+(\d{4})([a-z]?)$")


def parse_entry(s: str) -> str | None:
    """Normalise a single 'Author Year' tag (collapsing internal whitespace)."""
    s = re.sub(r"\s+", " ", s).strip()
    m = ENTRY_RE.match(s)
    if not m:
        return None
    author = m.group(1).strip()
    year = m.group(2)
    return f"{author} {year}"


def split_group(inside: str) -> list[str] | None:
    """Try to split a [...] inside-text into multiple author-year tags.

    Returns the list of canonical tags if every entry parses successfully,
    or None if the bracket isn't a citation (e.g. a numeric range).
    """
    parts = re.split(r"[,;]", inside.replace("\n", " "))
    out = []
    for p in parts:
        tag = parse_entry(p)
        if tag is None:
            return None
        out.append(ALIAS.get(tag, tag))
    return out


def find_citations(text: str) -> list[str]:
    """Return canonical tags in order of first appearance in body text only."""
    refs_idx = text.rfind("## References")
    body = text[:refs_idx] if refs_idx > 0 else text
    out = []
    seen = set()
    for m in GROUP_RE.finditer(body):
        tags = split_group(m.group(1))
        if tags is None:
            continue
        for t in tags:
            if t in seen:
                continue
            if t not in REF_TEXT:
                print(f"  WARNING: unknown citation '{t}' -- skipping")
                continue
            seen.add(t)
            out.append(t)
    return out


def renumber(text: str, dry: bool = False) -> str:
    appearance_order = find_citations(text)
    num_for = {tag: i + 1 for i, tag in enumerate(appearance_order)}
    print(f"\n{len(num_for)} unique citations in appearance order:")
    for tag, n in num_for.items():
        print(f"  [{n:>2}] {tag}")
    if dry:
        return text

    refs_idx = text.rfind("## References")
    body = text[:refs_idx]
    after_refs = text[refs_idx:]

    def sub(m):
        tags = split_group(m.group(1))
        if tags is None:
            return m.group(0)
        nums = []
        for t in tags:
            if t in num_for:
                nums.append(num_for[t])
            else:
                # Unknown — leave the bracket alone.
                return m.group(0)
        if not nums:
            return m.group(0)
        # Sort and collapse contiguous runs into ranges (Vancouver style).
        nums = sorted(set(nums))
        parts = []
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if i == j:
                parts.append(str(nums[i]))
            elif j == i + 1:
                parts.append(f"{nums[i]},{nums[j]}")
            else:
                parts.append(f"{nums[i]}-{nums[j]}")
            i = j + 1
        return "[" + ",".join(parts) + "]"
    body = GROUP_RE.sub(sub, body)

    new_refs = ["## References", ""]
    new_refs.append(
        "(Camera-ready BibTeX is provided as `refs.bib`. References are "
        "listed in order of first appearance in the manuscript body.)"
    )
    new_refs.append("")
    for tag in appearance_order:
        new_refs.append(f"[{num_for[tag]}] {REF_TEXT[tag]}")
        new_refs.append("")

    return body + "\n".join(new_refs)


def main():
    dry = "--dry" in sys.argv
    text = SRC.read_text(encoding="utf-8")
    new_text = renumber(text, dry=dry)
    if not dry:
        bak = SRC.with_suffix(SRC.suffix + ".bak")
        bak.write_text(text, encoding="utf-8")
        SRC.write_text(new_text, encoding="utf-8")
        print(f"\nwrote {SRC} (backup at {bak})")


if __name__ == "__main__":
    main()
