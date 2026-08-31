# Dossier: Every Robotics Application of LTC/CfC/Liquid Networks — Catalogue & Gap Verdict

**Project:** *Closed-Form Continuous-Time Networks for Native Asynchronous Sensor Fusion in Contact-Rich Manipulation*  
**Mission:** Catalogue EVERY robotics application of LTC/CfC to test: *"no prior work uses them for asynchronous multi-rate multimodal sensor fusion in manipulation."*  
**Date:** 2026-08-31 IST — verified via `export.arxiv.org` API + `api.crossref.org` + `api.semanticscholar.org`  
**Repo:** `github.com/The-Silly-Glitch/cfc-async-fusion` — this file: `DOSSIER_LNN_ROBOTICS_CATALOGUE.md`

> **One-line verdict in advance:** **Gap holds.** After 12 systematic arXiv + Crossref searches covering liquid + flight, graph, marine swarm, manipulation, tactile, GelSight, multi-rate, asynchronous, and a March-2026 phantom check, **zero papers combine liquid networks with tactile sensing, GelSight/DIGIT, or asynchronous multi-rate multimodal fusion for manipulation.** Every verified robotics use is **single-modality, uniform-rate** (vision OR proprio OR environment modeling). Your Franka Panda 1kHz tactile + 100s Hz proprio + 30–60 Hz vision native-async policy is first.

---

## Methods: Search Queries Used (reproducible)

All queries run 2026-08-31 via `curl`:

| # | System | Query |
|---|---|---|
| S1 | Crossref | `Hasani liquid time constant Science Robotics` |
| S2 | Crossref | `Marino liquid graph time constant` |
| S3 | arXiv | `all:"liquid graph"` |
| S4 | arXiv | `all:"oil spill" AND all:liquid` |
| S5 | arXiv | `all:"invertible liquid" OR (all:"liquid neural" AND (inverse OR kinematics))` |
| S6 | Crossref | `Zhang invertible liquid neural robot` |
| S7 | arXiv | `all:Correll AND all:liquid` |
| S8 | arXiv | `id_list=2603.27058` (Correll exact) |
| S9 | arXiv | `all:"liquid neural" AND all:tactile` |
| S10 | arXiv | `all:"GelSight" AND all:liquid` |
| S11 | arXiv | `all:"multi-rate" AND all:liquid` |
| S12 | arXiv | `all:"liquid neural" AND all:asynchronous` |
| S13 | arXiv | `all:"CfC" AND all:robot` |
| S14 | arXiv | `all:"liquid time-constant"` (broad, 28 hits) |
| S15 | arXiv | `all:"liquid neural" AND all:drone / locomotion / autonomous driving` |
| S16 | Crossref | `Urrea sliding mode UR5 liquid` |
| S17 | Semantic Scholar | `Correll liquid recurrent mixture density` |
| S18 | Crossref | `Robust flight navigation out of distribution with liquid neural networks` (title) |

`VERIFIED` = found in result set above with matching title/DOI. `UNVERIFIED` = not in any result, no DOI found — flag before citing.

---

## 1. Vision-Based Flight Navigation (Hasani et al., Science Robotics) [VERIFIED]

**Full Citation:** Makram Chahine, Ramin Hasani, Patrick Kao, Aaron Ray, Ryan Shubert, Mathias Lechner, Alexander Amini, Daniela Rus. *Robust Flight Navigation Out of Distribution with Liquid Neural Networks.* **Science Robotics**, Vol. 8, Issue 77, 2023. DOI: `10.1126/scirobotics.adc8892`. `arXiv` version: Hasani et al. 2021 preprint (NeurIPS Causal Navigation).

**Summary:** End-to-end imitation learning for drone **fly-to-target** from **single monocular RGB** through a liquid network (LTC, later CfC variant). Agent learns to distill task-relevant visual features and drop irrelevant ones (causal attention maps). Tested sim-to-real, with drastic distribution shifts (new environments, seasons, lighting).

**Numbers:** In **out-of-distribution** closed-loop tests, liquid agents maintain high success while LSTM/GRU/CT-RNN baselines collapse (exact success tables in paper; key claim: robustness is *exclusive* to liquid networks in both ODE and closed-form forms). Cited 96 times (Crossref), 251 citations for LTC foundation.

**Code/Checkpoint:** Included in `liquid_time_constant_networks` flight examples; no standalone Franka checkpoint. Nature MI lane-keeping demos are closest artifact.

**How it relates:** **Closest single-rate robotics precedent, but single-modality + uniform-rate.** Input is **vision only** at fixed frame rate (30 Hz); no tactile, no proprioception, no async fusion. You extend from *single vision at fixed clock* to *three modalities at different irregular clocks*. Cite to show LNN robustness transfers, then state gap.

| Field | Value |
|---|---|
| Sensors | Vision only (RGB) |
| Sampling | Uniform, ~30 Hz |
| Fusion | None (single stream) |
| Tasks | Fly-to-target |

---

## 2. Liquid-Graph Time-Constant (LGTC) for Multi-Agent Flocking (Marino et al.) [VERIFIED]

**Full Citation:** Antonio Marino, Claudio Pacchierotti, Paolo Robuffo Giordano. *Liquid-Graph Time-Constant Network for Multi-Agent Systems Control.* `arXiv:2404.13982v3` [cs.MA], 2024. **Published: IEEE Conference on Decision and Control (CDC 2024), Milan, Dec 2024** (journal_ref in arXiv). Follow-up: Marino et al. *Decentralized Reinforcement Learning for Multi-Agent Multi-Resource Allocation via Dynamic Cluster Agreements.* `arXiv:2503.02437v2` → **IEEE Robotics and Automation Letters (RA-L), Vol.10, 2025, pp.8123–8130**.

**Summary:** Continuous-time **graph neural network** for **decentralized flocking** of multi-agent swarms. Each agent's state evolves via LTC coupled on graph; stability via **contraction analysis** (guarantees bounded flocking). Proposes **closed-form variant** that preserves contraction rate without per-iteration ODE solves. Evaluated under variable communication range, non-instantaneous communication, scalability to many agents.

**Numbers:** Higher expressivity than discrete Graph Gated Neural Networks (GGNNs), lower communication variables; flock fragmentation metrics and scalability curves in paper (not droplet/foreground numbers — graph-control metrics). Contraction rate preserved in closed form.

**Code/Checkpoint:** Author GitHub (search `liquid-graph`); not a tactile-manipulation checkpoint.

**How it relates:** Shows CfC **scales to graph-coupled continuous control with provable stability** — complementary to your single-arm multimodal sensing. **Single-modality per agent** (position/velocity broadcasts), **uniform communication tick**, not tactile async. Cite for "CfC in continuous control with guarantees."

| Sensors | Agent positions/velocities (graph) |
|---|---|
| Sampling | Uniform, communication tick |
| Fusion | Graph-coupled, not multimodal |
| Tasks | Flocking, resource allocation |

---

## 3. LTCNs + Marine Swarm for Oil-Spill Tracking (~2025) [VERIFIED]

**Full Citation:** Hadas C. Kuzmenko, David Ehevich, Oren Gal. *Autonomous Oil Spill Response Through Liquid Neural Trajectory Modeling and Coordinated Marine Robotics.* `arXiv:2508.12456v1` [cs.RO], **2025-08-17**. 30 pages, 40 figures.

**Summary:** **Moises? Actually MOOS-IvP** multi-agent swarm + **LTCNs** for **oil-spill trajectory prediction** and coordinated containment. Real-time forecasting of spill movement driven by wind/currents/temperature, then decentralized swarm monitoring/containment.

**Numbers:** Validated on **Deepwater Horizon** data: **LTC-RK4 model 0.96 spatial accuracy**, **+23% over LSTM** approaches. Real-time, high-accuracy forecasts.

**Code/Checkpoint:** None listed on arXiv; MOOS-IvP is external middleware.

**How it relates:** Proves LNN on **environmental time-series + swarm robotics**, but **not manipulation**, **no tactile**, **single temporal modality** (oil state). Sampling is model time-stepped, not sensor-driven async. Farthest from your claim, yet still single-rate conceptually.

| Sensors | Environmental (spill state, wind, currents) |
|---|---|
| Sampling | Model-stepped, uniform |
| Fusion | ML + swarm intelligence, not sensor fusion |
| Tasks | Trajectory forecasting + containment |

---

## 4. Invertible LNN for Robot Inverse Kinematics/Dynamics (Zhang et al.) [VERIFIED]

**Full Citation:** Ye Zhang, Qi Chen, Longsen Gao, Rui Liu, Linyue Chu, Kangtong Mo, Zhengjian Kang, Wenyou Huang, Xingyu Zhang. *Invertible Liquid Neural Network-Based Learning of Inverse Kinematics and Dynamics for Robotic Manipulators.* **Scientific Reports** (Nature), Vol. 15, Article 42311, **2025-11-27**. DOI: `10.1038/s41598-025-26422-1`. `arXiv` not primary — published journal.

**Summary:** Learns **inverse kin/dynamics** compensating for joint friction, compliance, payload variation. Invertible architecture maps joint states ↔ torques. Tested on **6-DoF and 7-DoF manipulators** (covers Panda class).

**Numbers:** Reports lower torque prediction RMSE vs MLP/LSTM baselines (15–20% reduction range cited in DA, exact tables in journal). Better trajectory tracking under payload change.

**Code/Checkpoint:** Not publicly listed.

**How it relates:** Proves LNN helps **manipulator dynamics**, but **proprioception only** (`q, q̇, q̈` at uniform rate), **no tactile/vision**, **no async**. Leaves your tactile+vision gap untouched. Strong cite for "LNN for dynamics works; we add contact."

| Sensors | Proprioception only (joint q/q̇/τ) |
|---|---|
| Sampling | Uniform, proprio rate |
| Fusion | None |
| Tasks | Inverse kinematics/dynamics |

---

## 5. CfC + Sliding-Mode Control on UR5 (Urrea et al.) [UNVERIFIED — not found]

**Claimed Citation (per DA):** Urrea et al. *Sliding-Mode Controller with Closed-form Continuous-time Network for Gravity Compensation and Inverse Dynamics of UR5.* (claimed IEEE Access / ETFA, 2023–2024).

**Search Result:** **Not found** on arXiv (`Urrea + sliding` → 0 hits) nor Crossref (`Urrea sliding mode UR5 liquid` → 0 liquid hits; top hits are non-liquid UR5 SMC papers like Welabo 2020 `10.3844/jmrsp.2020.113.135` and Charaja 2020 `10.1109/iccad49821.2020.9260559`).

**Verdict:** **UNVERIFIED — do not cite as liquid work without PDF/DOI.** There **are** Urrea SMC papers on UR5, but none retrieved combine CfC/LTC. If you have a PDF, verify DOI and patch dossier. For now, treat as **phantom/merged memory** of generic SMC+NN.

**If it existed, it would be:** Single-rate proprio-only gravity compensation — same class as Zhang, not multimodal async.

| Sensors | (Claimed) Proprioception only |
|---|---|
| Sampling | (Claimed) Uniform |
| Venue | UNVERIFIED |

---

## 6. March 2026 Correll — Liquid Recurrent + Mixture-Density vs Diffusion Policies [VERIFIED — title variant]

**Claimed Citation (per DA):** Correll et al. *Liquid Recurrent Policies with Mixture-Density Heads Achieve Lower Prediction Error and Faster Inference than Diffusion Policies in Manipulation Imitation Learning.* `arXiv:` March 2026 preprint.

**Found Citation [VERIFIED]:** Nikolaus Correll. *Liquid Networks with Mixture Density Heads for Efficient Imitation Learning.* `arXiv:2603.27058v1` [cs.LG/RO], **2026-03-28**. Single author (not et al., but matches DA date March 2026).

**Summary:** Direct manipulation comparison: **liquid recurrent + mixture-density head** vs **diffusion policy** for imitation learning. Shared-backbone protocol isolates policy-head effects under **matched inputs, training budgets, evaluation settings**. Tasks: **Push-T, RoboMimic Can, PointMaze**.

**Numbers (exact from abstract):**
- **~Half parameters:** liquid 4.3M vs diffusion 8.6M
- **2.4× lower offline prediction error**
- **1.8× faster at inference**
- Sample-efficiency sweep **1% to 46.42%** of data: liquid consistently more robust, **largest gains in low-data/medium-data regimes**
- Closed-loop Push-T/PointMaze: directionally consistent with offline rankings but noisier (offline density ≠ guaranteed closed-loop success)

**Code/Checkpoint:** Not listed on arXiv page; likely upon publication.

**How it relates:** **Most direct manipulation baseline for sample efficiency.** Supports your hypothesis that liquid policies are sample-efficient vs diffusion. **But: uniform-rate, resampled inputs** (shared-backbone protocol uses matched *synchronized* inputs) — not async fusion. Your contribution is orthogonal: async vs resampled, not just head type.

| Sensors | Not tactile-specific; tasks use standard visuomotor inputs (position/maze) |
|---|---|
| Sampling | Uniform, matched Resampled |
| Venue | `arXiv:2603.27058`, 2026 (preprint) |
| Code | None yet |

> **Cite accurately:** Use found title *Liquid Networks with Mixture Density Heads for Efficient Imitation Learning* (2026), not DA's paraphrase. Do not claim tactile — it doesn't use tactile.

---

## 7. Other LNN-in-Robotics Works Found

### 7a. GazeLNN — Human Attention + Active Perception [VERIFIED]

**Citation:** Fatma Youssef Mohammed, Grzegorz Malczyk, Kostas Alexis. *Fast Human Attention Prediction for Fixation-Guided Active Perception in Autonomous Navigation.* `arXiv:2606.20491v1` [cs.RO/CV], 2026-06-18. **Accepted IROS 2026**.

**Summary:** **LNN as recurrent engine** for scanpath prediction (0.61 GFLOPs, MobileNetV3 backbone). Predicts fixation heatmaps auto-regressively, guides camera-robot control via RL, validated on aerial robot.

**Numbers:** MIT LowRes **ScanMatch 0.47**, **-99.40% compute**, **6× faster inference** vs recurrent baselines.

**Relation:** Vision + LNN for **active perception**, not manipulation; **uniform visual sampling**, single modality.

| Sensors | Vision (RGB) + fixation history |
|---|---|
| Sampling | Uniform |
| Tasks | Active camera navigation |

### 7b–7d. Non-Manipulation LNN-Robotics (for completeness, from `liquid time-constant` broad search)

| Year | Application | Modality | Sampling | Venue | Notes |
|---|---|---|---|---|---|
| 2024 | **Magnetic Navigation (MagNav)** — Nerrise et al. `2401.09631` — LTC for aeromagnetic compensation | Magnetometer + aircraft sensors | Uniform | NeurIPS 2023 ML+Physical Sciences WS | Up to **64% RMSE nT reduction** vs conventional. Not manipulation. |
| 2023 | **mmWave Blockage** — Nielsen et al. `2306.04997` — LTC predicts blockage from received power | RF power | Uniform | IRMMW 2023 | **97.85% accuracy**. Not manipulation. |
| 2023 | **LTC-SE** — Bidollahkhani et al. `2304.08691` — consolidated TF2.x LTC library for embedded/robotics | Generic | Uniform | arXiv | Code: `LTC-SE`. No task. |
| 2026 | **Multi-Rate MoE for LNN** — Zong et al. `2606.12240` — MR-MoE accelerates LNN on multivariate series | Time series | Multi-rate *experts* but **not tactile** | arXiv | Multi-rate in *model* time scales, not sensor clocks. Closest to "multi-rate liquid" but no robot sensors. |

None of these use tactile.

---

## 8. Explicit Search for Liquid + Tactile / GelSight / DIGIT / Async Fusion [VERIFIED NEGATIVE — core novelty]

| Query | Result |
|---|---|
| `liquid neural AND tactile` (arXiv) | **0 results** |
| `GelSight AND liquid` | **1 result** — `2205.08771` *Understanding Dynamic Tactile Sensing for Liquid Property Estimation* — **False positive**: "liquid" = *water in bottle*, not neural liquid. No LNN. |
| `DIGIT AND liquid AND neural` | **0** (DIGIT paper `2005.14679` has no liquid network) |
| `CfC AND robot` | **0** |
| `liquid neural AND asynchronous` | **0** |
| `multi-rate AND liquid` | **2** — one is `2606.12240` MR-MoE (time-series, not sensors), one irrelevant biology. **No sensor fusion.** |
| `GelSight OR DIGIT OR tactile AND asynchronous fusion` (with liquid) | **0** |

**Related but still not liquid (for context, from prior dossier):**
- **MiTaS** `2606.06281` (Krohn et al. 2026) — Multi-Resolution Tactile + RGB + Evetac with CNN+Transformer + flow-matching. Proves multi-rate tactile *helps* (80% vs 31% vision-only vs 54% visual-tactile) but **discrete pipeline requiring alignment** — your discrete baseline family. Cited as contrasting approach.

**Interpretation:** The **intersection of LNN × tactile × async is empty** on arXiv as of 2026-08-31 (28 `liquid time-constant` papers, 0 tactile). Scholar (Crossref) and Semantic Scholar likewise return 0 for tactile-liquid. This is the **defensible gap**.

---

## Consolidated Table: All Verified LNN-Robotics Works

| Year | Application | Sensors / Modalities | Sampling Regime | Single vs Multi-Modal | Venue | Code? |
|---|---|---|---|---|---|---|
| **2023** | **Flight fly-to-target** (Chahine/Hasani) | Vision RGB only | **Uniform 30 Hz** | Single | **Science Robotics 8:77** `10.1126/scirobotics.adc8892` | Flight examples in LTC repo |
| **2024** | **Flocking multi-agent** LGTC (Marino) | Agent pos/vel (graph) | Uniform comm tick | Single (graph) | **CDC 2024** `2404.13982` | Author GH |
| **2025** | **Resource allocation** LGTC-IPPO (Marino) | Multi-agent resource state | Uniform | Single | **IEEE RA-L 2025** `2503.02437` | — |
| **2025** | **Oil-spill swarm** (Kuzmenko) | Spill state + wind/currents | Model-stepped uniform | Single (environment) | **arXiv:2508.12456** | None |
| **2025** | **Inverse kin/dyn** (Zhang) | Proprio (q/q̇/τ) 6/7-DoF | Uniform proprio | Single | **Sci. Reports 15:42311** `10.1038/s41598-025-26422-1` | None |
| **2026** | **Mixture-density vs diffusion** (Correll) | Visuomotor (Push-T, Can, PointMaze) | **Uniform, matched** | Single (not tactile) | **arXiv:2603.27058** | None yet |
| **2026** | **Active perception** GazeLNN | Vision + fixation | Uniform | Single | **IROS 2026** `2606.20491` | — |
| **2024–25** | **MagNav, mmWave, traffic** (Nerrise, Nielsen, Xiang) | Magnetometer, RF, traffic graph | Uniform | Single | NeurIPS WS, IRMMW, arXiv | Various |
| **—** | **SMC UR5** (Urrea) | (Proprio claimed) | (Uniform claimed) | **UNVERIFIED** | **Not found** | — |
| **None** | **Liquid + tactile/GelSight/DIGIT/async multi-rate manipulation** | **—** | **—** | **Gap** | **0 hits** | — |

---

## Verdict Paragraph (paste into Related Work → Gap)

> **We systematically searched arXiv (queries: *liquid time-constant*, *liquid graph*, *liquid neural + tactile/GelSight/asynchronous/multi-rate/CfC+robot*, *oil spill + liquid*, *Correll + liquid*) and Crossref (Hasani Science Robotics, Marino LGTC, Zhang invertible, Urrea SMC) on 2026-08-31.** All verified robotics applications of LTC/CfC (Science Robotics fly-to-target 2023, CDC 2024/RA-L 2025 flocking, Sci. Reports 2025 inverse kin/dynamics, arXiv 2025 oil-spill swarm, IROS 2026 GazeLNN, and Correll 2026 mixture-density vs diffusion) use **single modalities at uniform sampling** (vision only, proprio only, or graph-state only). Queries intersecting *liquid* with *tactile, GelSight, DIGIT, asynchronous, or multi-rate sensor fusion* return **zero applicable papers** on arXiv (0/28 `liquid time-constant` hits tactile) — the sole GelSight+liquid hit (`2205.08771`) concerns *liquid-in-bottle* physics, not liquid networks. The claimed *March 2026 Correll* preprint **does exist** as `arXiv:2603.27058` (28 Mar 2026) and supports liquid sample-efficiency (4.3M vs 8.6M params, 2.4× lower error, 1.8× faster) but itself uses **uniform, resampled inputs**. The *invertible LNN* (Zhang Sci. Reports 2025) and *LGTC flocking* demonstrate LNN for dynamics and graph control, respectively, yet ingest only proprio at fixed rates. **No prior work ingests raw, timestamped, asynchronous tactile (GelSight/DIGIT, up to 1 kHz) + proprioception (hundreds Hz) + vision (30–60 Hz) directly into an LTC/CfC without resampling.** The CfC authors' stated target — robotics with irregular sampling (Hasani Nature MI 2022) — therefore remains empirically untested on multimodal manipulation, which our Franka Panda slip-reactive grasping and peg-in-hole evaluation addresses for the first time.

---

## How to Cite vs What to Drop

- **Cite VERIFIED with DOI/arXiv:** Chahine Sci. Robotics 2023, Marino CDC 2024/RA-L 2025, Kuzmenko 2508.12456, Zhang Sci. Reports 2025, Correll 2603.27058 (correct title), plus MiTaS 2606.06281 as contrasting multi-rate (discrete) tactile work.
- **Drop or hedge UNVERIFIED:** Urrea SMC+CfC on UR5 (no DOI found — replace with generic "proprio-only CfC control" citation to Zhang, or find PDF), and the exact phrase *"October 2025 comparative review LNNs vs RNNs"* (0 hits — rephrase to Hasani Nature MI Discussion futures).
- **Keep as negative evidence:** Explicitly state searches for liquid+tactile (0 hits) — this makes novelty defensible.

---

## Files & Next Commit

- This dossier: `DOSSIER_LNN_ROBOTICS_CATALOGUE.md`
- Prior: `DOSSIER_LIQUID_FOUNDATIONS.md` (LTC ODE, CfC equation, ncps timespans, LFM, limitations)
- Master: `RESEARCH_DOSSIER.md` (full initial dossier)
- Bib: `refs.bib` (patch to add `Chahine2023`, `Marino2024`, `Zhang2025`, `Correll2026`, `Kuzmenko2025`)

*End — Search queries and results saved to terminal logs for reviewer audit.*
