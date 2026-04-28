# Cover letter — *Journal of Materials Science & Technology*

Sunwoo Lee
Independent researcher
South Korea
ORCID: 0009-0004-9159-367X
Email: lee.11539@buckeyemail.osu.edu

April 2026

To the Editor-in-Chief,
*Journal of Materials Science & Technology*

Dear Editor,

I am pleased to submit the manuscript **"Cubic-Equivariant k-Space
Convolution and a Differentiable Bond-Valence Field for Ionic-
Conductivity Prediction in Solid-State Electrolytes"** for consideration
as an original research article in *JMST*.

## Why this work suits *JMST*

Solid-state lithium-ion batteries are a strategic technology for energy
storage, and predicting the room-temperature ionic conductivity σ of
solid-state electrolytes (SSEs) from crystal structure alone is a long-
standing bottleneck for materials discovery. *JMST*'s scope — bridging
fundamental materials science with technology-relevant innovation —
matches the contribution: two architectural primitives that are
methodologically novel, empirically the lowest neural MAE I obtain
on the OBELiX 281-sample filter, and immediately applicable to the
broader inorganic crystal-property literature.

## Key contributions

The manuscript introduces **two architectural primitives**: (i) k-SEC,
a reciprocal-space neural encoder that enforces cubic-group
equivariance by construction via cubic-harmonic directional filters
and cross-shell gated attention, and (ii) DBVF, an end-to-end-
learnable parameterisation of Brown's bond-valence model embedded
inside a neural network. Together they attain **MAE 1.047 standalone
and 0.980 in a stacked ensemble** on the OBELiX benchmark — the best
neural MAE I obtain on this filter and a 33 % reduction over
CGCNN-lite.

Beyond MAE, the headline ensemble achieves **Spearman ρ = 0.78 and
top-10 precision 70 %** (a 19.7× lift over random ranking) on the
OOF cross-validation set, and a 281-sample-trained virtual screen on
18,574 Materials Project crystals **independently identifies all four
known fast-Li conductor families** in its top-15. A controlled
architecture-vs-features experiment shows that the DBVF gain is
realised **only through end-to-end gradient flow** (the same DBVF
features fed to LightGBM do not help), supporting an architectural
rather than feature-engineering interpretation of the contribution.

## What I am honest about

- A gradient-boosted tree on hand-crafted features (Magpie + lattice +
  geometric) achieves MAE 0.924, lower than my standalone neural
  number (1.047). I frame this as the well-documented small-data
  ceiling [Grinsztajn 2022, McElfresh 2023] rather than a fundamental
  architectural failure.
- Four other architectural extensions (heteroscedastic loss,
  percolation features, path-integrated DBVF, dual-stream BatteryNet)
  were tried and *did not* improve the headline; I document each
  negative result in the SI rather than burying them.
- The Li-pretrained encoder does **not** transfer cleanly to the
  Matbench elasticity target, indicating domain-specific rather than
  universal transferability.

I believe this combination — a methodologically novel architecture,
honest negative results, and a clean computational-validation story on
real fast-conductor families — is exactly the kind of contribution
*JMST* readers value.

## Suitability and originality

The manuscript has not been published or submitted elsewhere. I have
read and approved the submission and declare no competing interests.
Primary code, trained weights, OBELiX-derived data artifacts, and the
full virtual-screening output are released at the GitHub repository
accompanying the submission; the headline result is reproducible in
~6 h on a single 12-GB consumer GPU.

I thank you for considering my work and look forward to your
editorial assessment.

Sincerely,

Sunwoo Lee
Independent researcher, South Korea
ORCID: 0009-0004-9159-367X
lee.11539@buckeyemail.osu.edu

## Suggested reviewers (per *JMST* policy, ≥4)

1. **Prof. Shyue Ping Ong** — Department of NanoEngineering,
   University of California, San Diego, USA. Materials informatics,
   M3GNet, MEGNet. Group page: https://materialsvirtuallab.org/.
2. **Dr. Kamal Choudhary** — National Institute of Standards and
   Technology, Gaithersburg, MD, USA. ALIGNN, JARVIS-DFT.
   Profile: https://www.nist.gov/people/kamal-choudhary.
3. **Prof. Yifei Mo** — Department of Materials Science and
   Engineering, University of Maryland, College Park, USA.
   Solid-state Li-ion electrolytes, AIMD diffusion modelling.
   Group page: https://yifeimo.umd.edu/.
4. **Prof. Stefan Adams** — Department of Materials Science and
   Engineering, National University of Singapore. Bond-valence
   pathway analysis for ionic conductors.
   Profile: https://cde.nus.edu.sg/mse/staff/adams-stefan/.
5. **Dr. Ilyes Batatia** — Department of Engineering, University of
   Cambridge, UK. MACE / MACE-MP-0 equivariant interatomic
   potentials. Profile: https://ilyes.batatia.eu/.

## Non-preferred reviewers

None.
