# CfC Async Sensor Fusion — Research Dossier

**Paper:** *Closed-Form Continuous-Time Networks for Native Asynchronous Sensor Fusion in Contact-Rich Manipulation*  
**Authors (DA):** Nahul Alaguraj, Akhilesh Manchi, Hanish Vigneshwar  
**Folder:** `/home/honeysh/Projects/cfc-async-fusion/` (also mirrored to `CfC-Async-Sensor-Fusion/`)

## What's inside

| File | Purpose |
|---|---|
| `RESEARCH_DOSSIER.md` | Full structured dossier — every paper with (1) citation (2) summary (3) numbers (4) code/checkpoint (5) relation to your paper. Flags VERIFIED vs UNVERIFIED. |
| `refs.bib` | BibTeX for all VERIFIED papers — paste into Overleaf. UNVERIFIED entries marked. |
| `references/` | Drop PDFs here (`PAPERS/` alias). Suggested: LTC, CfC, MiTaS, Taxim, DIGIT, Diffusion Policy. |

## Quick start — reproduce the backbone

```bash
pip install ncps torch  # NCPS = https://github.com/mlech26l/ncps
# Example: CfC that consumes irregular delta_t natively
from ncps.torch import CfC
from ncps.wirings import AutoNCP

wiring = AutoNCP(32, 8)  # 32 neurons, 8 outputs — tune to match LSTM param count
cfc = CfC(input_dim=64, wiring=wiring)  # 64 = vision_emb + tactile_emb + proprio_emb
# forward: out, new_state = cfc(input_emb, hidden_state, timespan=delta_t)
# delta_t is the actual elapsed time since last sensor event — no resampling
```

## How to verify UNVERIFIED citations before submission

```bash
# Gateway `nous` was down during generation; use direct arXiv API:
curl "https://export.arxiv.org/api/query?id_list=2006.04439,2106.13898,2005.14679,2606.06281,2303.04137,1706.03762,1803.01271,2109.04027"
# For papers not on arXiv (Science Robotics, LGTC, Zhang UR5), search IEEE Xplore + CrossRef by title.
```

## Next actions

1. `curl` the 4 verified PDFs into `references/` (or `PAPERS/`).
2. Confirm `B1–B5` (flight, LGTC, Zhang, Urrea, Correll 2026) in IEEE/ICRA — remove if not found.
3. Implement 3 baselines equal-parameter: `nn.LSTM` (ZOH), `locuslab/TCN`, `nn.Transformer` on resampled 60Hz.
4. Franka Panda tasks: slip-reactive grasping + peg-in-hole in MuJoCo Menagerie / Isaac Sim.

---
Generated 2026-08-31 IST — patched with live arXiv verification (DIGIT corrected to 2005.14679).
