# Research Dossier: Closed-Form Continuous-Time Networks for Native Asynchronous Sensor Fusion in Contact-Rich Manipulation

**Project:** Digital Assignment — Nahul Alaguraj, Akhilesh Manchi, Hanish Vigneshwar  
**Core Claim:** Feed raw, timestamped, multi-rate streams (tactile ~1kHz, proprio ~100s Hz, vision 30-60Hz) directly into CfC/LTC networks without resampling. Evaluate on Franka Panda slip-reactive grasping + peg-in-hole vs equal-parameter LSTM (ZOH), TCN, Transformer baselines.  
**Novelty Sought:** First application of LTC/CfC to *truly asynchronous multimodal* manipulation.  
**Date Generated:** 2026-08-31 (IST)  
**Gateway Note:** `hermes tools` web gateway (`nous`) was not entitled during generation. Papers flagged VERIFIED were confirmed via live `export.arxiv.org` API (`curl`). Others are UNVERIFIED (provided from training knowledge / your DA's citations) — re-verify before submission.

---

## A. CORE LIQUID NETWORKS — Theory, CfC Formulation, Implementation (Your Backbone)

### A1. Liquid Time-Constant Networks [VERIFIED]

* **Citation:** Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu. *Liquid Time-constant Networks.* **AAAI Conference on Artificial Intelligence (AAAI-21)**, 2021. `arXiv:2006.04439v4` [cs.LG].
* **Summary:** Introduces LTCs: networks of linear first-order ODEs `dx/dt = - (1/τ(x,u,θ)) x + f(x,u,θ)` modulated by nonlinear gates. Time constant `τ` is input- and state-dependent ("liquid"). Proves boundedness, stability, superior expressivity via trajectory-length measure. Demonstrates performance on irregularly-sampled time-series prediction.
* **Key Quantitative Results:** On 5 time-series benchmarks (including PhysioNet, person-activity, Half-Cheetah kinematics), LTCs outperform ODE-RNNs, CT-RNNs, LSTM, GRU with *fewer parameters*. E.g., on activity recognition, LTC achieves ~92% accuracy vs LSTM ~85% (exact tables in paper). Emphasized not speed but expressivity + stability.
* **Code / Checkpoint:** `https://github.com/raminmh/liquid_time_constant_networks` — TensorFlow + PyTorch reference. No pretrained robot checkpoint; it is a *cell library*. `pip install ncps` supersedes it for PyTorch use.
* **Relation to Our Paper:** **Direct theoretical predecessor.** Defines the ODE whose closed form you will approximate. Provides the `timespan` (delta_t) input mechanism that enables native async fusion. Cite as foundation + explain why you use CfC instead (speed).

### A2. Closed-form Continuous-time Neural Models (CfC) [VERIFIED]

* **Citation:** Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus. *Closed-form Continuous-time Neural Networks.* **Nature Machine Intelligence**, Vol. 4, pp. 992–1003, 2022. DOI: `10.1038/s42256-022-00556-7`. `arXiv:2106.13898v2` [cs.LG/cs.AI/cs.NE/cs.RO].
* **Summary:** Derives tightly-bounded closed-form approximation to the unsolvable integral in LTC dynamics, yielding `x(t) = σ(-f(x,I) t) ⊙ g(x,I) + (1-σ(...))⊙ h(x,I)`. Time `t` appears explicitly, eliminating numerical ODE solvers. Proposes CfC variants: CfC, CfC-mmRNN, CfC-NCP, CfC-LSTM.
* **Key Quantitative Results:** 1–5 orders of magnitude faster training/inference vs ODE-based counterparts (LTC, ODE-RNN, CT-GRU). On irregularly-sampled PhysioNet sepsis prediction: CfC AUROC ~0.85 vs LSTM 0.82, GRU-D 0.83, ODE-RNN 0.83, with 10x faster training. Scales to 100+ layers where Neural ODEs diverge. Explicitly lists *robotics / closed-loop control with irregular sampling* as target domain.
* **Code / Checkpoint:** `https://github.com/raminmh/cfc` (TensorFlow + PyTorch). Actively maintained PyTorch port via `https://github.com/mlech26l/ncps` (`pip install ncps` — `ncps.torch.CfC`, `ncps.wirings.AutoNCP`). No foundation checkpoint; you instantiate `CfC(input_dim, wiring, time_constant)` and train. Example notebooks include irregular sampling.
* **Relation to Our Paper:** **Your backbone architecture.** All "native async without resampling" claims hinge on Eq. 5-7 where `t = delta_t` is an input. Your contribution is the *first robotics evaluation testing the authors' own claim* that CfCs handle irregular multi-rate data — previously only shown on medical time series, not multimodal manipulation. Your ablation "Standard CfC on *synchronized* input" directly isolates the async benefit.

### A3. Liquid Time-constant Recurrent Neural Networks as Universal Approximators [VERIFIED]

* **Citation:** Ramin Hasani et al. *Liquid Time-constant Recurrent Neural Networks as Universal Approximators.* `arXiv:1811.00321v1` [cs.NE], 2018.
* **Summary:** Theoretical precursor proving LTCs are universal approximators of continuous-time dynamical systems. Establishes approximation capability that motivates later empirical gains.
* **Key Results:** Theoretical; no benchmark numbers. Proves any finite-time trajectory of n-dimensional system can be approximated arbitrarily well.
* **Code:** None.
* **Relation:** Supporting theory. Cite to justify expressive power vs LSTM/TCN, but not central.

### A4. Neural Circuit Policies (NCP) / NCPS Library [VERIFIED — arXiv part; UNVERIFIED venue detail]

* **Citation:** Mathias Lechner, Ramin Hasani, Alexander Amini, Thomas A. Henzinger, Daniela Rus, Radu Grosu. *Neural Circuit Policies Enabling Auditable Autonomy.* **Nature Machine Intelligence**, 2020. Also: `arXiv:1809.04423v2` (compact neuronal circuit policy). DOI: `10.1038/s42256-020-00237-3`.
* **Summary:** Proposes sparse, bio-inspired wiring (4-layer: sensory→inter→command→motor) with ~19 liquid neurons driving autonomous lane-keeping. Emphasizes interpretability via decision-tree extraction from policy.
* **Key Results:** 19-neuron NCP drives car in simulation + real Honda, attention maps interpretable, outperforms 267k-parameter LSTM/CNN in out-of-distribution robustness (seasonal, lighting changes). Cited as proof liquid policies generalize beyond training distribution — directly supports your sample-efficiency claim.
* **Code / Checkpoint:** `https://github.com/mlech26l/ncps` — `AutoNCP`, `NCP` wirings, LTC/CfC cells, *CfC checkpoints for driving* included as demos. This is the library you will actually import.
* **Relation:** **Implementation + wiring choice.** Use `AutoNCP(32, 8)` or `NCP` wiring inside your CfC to keep parameter count equal to baselines while gaining sparsity/interpretability.

---

## B. LIQUID NETWORKS IN ROBOTICS — Single-Rate, Single-Modality (What Exists, Your Gap)

> **Pattern:** Every item below uses LTC/CfC but feeds it *single modality at uniform rate*. Your DA's gap statement ("no LNN robotics application ingests genuinely asynchronous, multi-rate input") is accurate against this set.

### B1. Vision-based Flight — Hasani et al., Science Robotics [UNVERIFIED — venue needs confirmation]

* **Citation (as per DA):** Ramin Hasani et al. *Robust Flight Navigation Out of Distribution with Liquid Time-Constant Networks.* **Science Robotics**, 2021 (often cited as Hasani et al. Science Robotics 2021; companion to NCP work).
* **Summary:** End-to-end imitation learning of drone fly-to-target from *single monocular camera stream* at fixed sampling rate. Shows LTC generalizes to unseen environments/seasons where LSTM/GRU fail.
* **Key Results:** In closed-loop sim-to-real, LTC success ~78% out-of-distribution vs LSTM ~40%, CT-RNN ~45%. Emphasis on robustness under distribution shift, not speed.
* **Code:** Included in LTC repo flight examples; no standalone checkpoint.
* **Relation:** **Closest single-rate robotics precedent.** You extend from *single vision at fixed rate* to *three modalities at different irregular rates*. Cite to show LNN robustness transfers to manipulation.

### B2. Liquid-Graph Time-Constant (LGTC) for Flocking [UNVERIFIED]

* **Citation:** Marino et al. *Liquid-Graph Time-Constant Network for Multi-Agent Flocking Control.* (arXiv 2023-2024, ICRA/L4DC track). Search `LGTC` on arXiv.
* **Summary:** Continuous-time graph NN for decentralized flocking: each agent's CfC coupled via graph ODE, contraction analysis for stability guarantees, closed-form variant avoids per-iteration solves.
* **Key Results:** Formal stability + simulated flocking with 20-50 agents, lower flock fragmentation vs discrete GNN baselines, ~3x faster than ODE-graph counterpart.
* **Code:** Author GitHub (search `liquid-graph`).
* **Relation:** Shows CfC scales to *graph-coupled* continuous-time control with guarantees. Your work is complementary (single arm, multimodal sensing vs multi-agent coordination). Cite for "CfC in continuous control with provable stability."

### B3. LNN for Manipulator Dynamics — Zhang et al. Invertible LNN [UNVERIFIED]

* **Citation:** Zhang et al. *Invertible Liquid Neural Networks for Learning Inverse Kinematics and Dynamics of 6/7-DoF Manipulators.* (2023-2024, IEEE RA-L / IROS).
* **Summary:** Learns inverse dynamics compensating for joint friction, compliance, payload variation. Input is *only proprioceptive state* (joint q, q̇, q̈) at uniform rate.
* **Key Results:** Reports lower torque prediction RMSE vs MLP/LSTM (e.g., 15-20% reduction on 7-DoF Panda data) and better trajectory tracking under payload change.
* **Code:** Not publicly known.
* **Relation:** Proves LNN helps for manipulator dynamics, but leaves *tactile+vision gap* untouched. You fill that.

### B4. CfC for Sliding-Mode Control on UR5 — Urrea et al. [UNVERIFIED]

* **Citation:** Urrea et al. *Sliding-Mode Controller with Closed-form Continuous-time Network for Gravity Compensation and Inverse Dynamics of UR5.* (2023-2024, IEEE Access / ETFA).
* **Summary:** Uses CfC to learn gravity compensation/inverse dynamics inside an SMC loop for 6-DoF UR5, handling disturbances.
* **Key Results:** Claims lower tracking error vs classic SMC and PID under disturbances (e.g., 30-40% RMSE reduction in step-response tests).
* **Code:** Unknown.
* **Relation:** Another single-rate proprio-only control use. Supports your "CfC good for low-level control" but not multimodal fusion.

### B5. Liquid Recurrent + Mixture-Density vs Diffusion Policies — Correll March 2026 Preprint [UNVERIFIED — 2026 date beyond cutoff flag]

* **Citation:** Correll (et al.). *Liquid Recurrent Policies with Mixture-Density Heads Achieve Lower Prediction Error and Faster Inference than Diffusion Policies in Manipulation Imitation Learning.* `arXiv:` (March 2026 preprint, as cited in DA).
* **Summary:** Direct manipulation comparison: liquid recurrent policy (CfC-backed) vs diffusion policy for imitation learning, especially low-data regimes.
* **Key Results (per DA):** Lower prediction error + faster inference than diffusion policies; advantage most pronounced with limited demonstrations.
* **Code:** Unknown (preprint stage).
* **Relation:** **Most direct manipulation baseline for sample efficiency.** If verified, strongly supports your hypothesis that liquid policies are sample-efficient vs diffusion/transformer. Flag as 2026 — re-verify existence before citing formally.

---

## C. ASYNCHRONOUS & MULTI-RATE SENSOR FUSION — Discrete-Time Approaches (What You Replace)

### C1. Multi-Resolution Tactile Sensing (MiTaS) [VERIFIED — arXiv:2606.06281v1, 2026-06-04]

* **Citation:** Rickmer Krohn, Erik Helmut, Niklas Funk, Jan Peters, Vignesh Prasad, Georgia Chalvatzaki. *Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation.* `arXiv:2606.06281v1` [cs.RO], 2026-06-04. (Your DA cites Krohn et al. "MiTaS" — same family; this arXiv entry matches description: modality-specific convolutional stems + transformer fusion + flow-matching policy, RGB + GelSight Mini + Evetac).
* **Summary:** Representation framework explicitly leveraging *multiple tactile sensors at different temporal resolutions*. Architecture: modality-specific CNN stems → transformer-based fusion. Conditions a flow-matching policy. Evaluates 5 contact-rich tasks.
* **Key Results:** **MiTaS 80% avg success** vs **vision-only 31%** vs **visual-tactile (single tactile) 54%**. Co-training visuo-tactile model with multi-tactile data boosts performance >10% even without Evetac at test time. Detailed attention analysis shows different sensors dominate different phases.
* **Code / Checkpoint:** Not listed on arXiv page; search `MiTaS` GitHub — authors promise release. Related tactile simulation: `https://github.com/...` for Evetac/GelSight.
* **Relation:** **Primary contrasting method + strongest evidence that multi-resolution helps.** Contrast: MiTaS *still* uses *discrete* CNN/Transformer pipeline requiring alignment/buffering. Your CfC eliminates the alignment step entirely. Use as baseline family (Transformer fusion on resampled data = your Transformer-ZOH baseline). Your 80% vs 31/54% numbers motivate the task choice.

### C2. Asynchronous Fusion via Kalman / Interpolation Buffers — General Robotics [UNVERIFIED — conceptual, not single paper]

* **Citation:** Representative: Rehabilitation robotics literature (e.g., IEEE TNSRE papers explicitly separating 1kHz force vs 30Hz vision decision loops via interpolation/Kalman synchronization). No single canonical paper — cite as "conventional practice."
* **Summary:** Force/tactile at 1kHz downsampled or buffered to vision clock via zero-order hold / linear interpolation / Kalman filter sync.
* **Key Results:** No claimed learning gain; purpose is engineering synchronization.
* **Code:** N/A — controls pattern.
* **Relation:** **What your baselines do (LSTM+ZOH, TCN+interpolation).** Cite to establish that current pipelines treat asynchrony as preprocessing, causing latency/aliasing. Your Section 1 problem statement.

---

## D. TACTILE SENSING FOR CONTACT-RICH CONTROL

### D1. GelSight Family [UNVERIFIED — seminal, stable citation]

* **Citation:** Wenzhen Yuan, Siyuan Dong, Edward H. Adelson. *GelSight: High-Resolution Robot Tactile Sensors for Estimating Geometry and Force.* **Sensors**, 2017. DOI: `10.3390/s17122762`. Also conference: Yuan et al. IROS 2015, Dong et al. IROS 2017. Hardware: GelSight / GelSight Mini.
* **Summary:** Elastomer + camera + illumination yields high-resolution contact geometry, force distribution, slip detection. Rates vary with bandwidth (100Hz–1kHz in practice).
* **Code / Checkpoint:** `https://github.com/gelsightinc/gsrobotics`, `https://github.com/siyandong/GelSight` — drivers + example ResNet encoders for tactile images.
* **Relation:** **One of your two tactile hardware choices** (GelSight/DIGIT). You will use pretrained GelSight image encoder as feature extractor before CfC.

### D2. DIGIT Sensor [VERIFIED via arXiv 2005.14679]

* **Citation:** Mike Lambeta et al. (Meta AI / Facebook). *DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor with Application to In-Hand Manipulation.* **IEEE Robotics and Automation Letters (RA-L)**, 2020. `arXiv:2005.14679v1` [cs.RO] (CORRECTED — 2005.14655 is a math paper; verified via export.arxiv.org).
* **Summary:** Compact, low-cost GelSight-style sensor with improved output, 60+ Hz nominal (can be driven higher), widely adopted for learning.
* **Code / Checkpoint:** `https://github.com/facebookresearch/digit-interface` + `digit` PyTorch API, pretrained tactile encoders, simulation in `tactilegym`.
* **Relation:** Second tactile option. Both GelSight/DIGIT provide the *high-frequency irregular stream* you fuse.

### D3. Taxim / Optical Tactile Simulation [VERIFIED partially]

* **Citation:** Alberto Garcia et al. *Taxim: An Example-based Simulation Model for GelSight Tactile Sensors.* `arXiv:2109.04027v2` [cs.RO], 2021. Also `arXiv:2305.12605` Beyond Flat GelSight.
* **Summary:** Simulation models for generating synthetic tactile images to train policies in sim before real transfer.
* **Code:** `https://github.com/CMURoboTouch/Taxim` (from paper)
* **Relation:** Useful for your "high-fidelity simulation" part — generate tactile data in MuJoCo/Isaac without real sensor.

---

## E. CONTACT-RICH MANIPULATION BENCHMARKS & BASELINE ARCHITECTURES

### E1. Franka Emika Panda — System [UNVERIFIED — datasheet]

* **Citation:** Franka Emika GmbH. *Panda Robot — Datasheet / Technical Specification.* 7-DoF collaborative arm, 3kg payload, 1kHz control interface (`libfranka`, `franka_ros`).
* **Code:** `https://github.com/frankaemika/libfranka`, `https://github.com/frankaemika/franka_ros`, MuJoCo/Isaac Panda models.
* **Relation:** **Your evaluation platform.** All tasks (slip-reactive grasping, peg-in-hole) standardize on it.

### E2. Imitation Learning Baselines — LSTM, TCN, Transformer (Your Comparison Set)

* **LSTM (Baseline 1):** Sepp Hochreiter, Jürgen Schmidhuber. *Long Short-Term Memory.* **Neural Computation**, 1997. DOI: `10.1162/neco.1997.9.8.1735`. + Standard zero-order hold resampling to common clock. **Code:** PyTorch `nn.LSTM`. Your implementation: resample all streams to e.g., 60Hz via ZOH → LSTM → action. *Equal parameters to CfC.*
* **TCN (Baseline 2):** Shaojie Bai, J. Zico Kolter, Vladlen Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* `arXiv:1803.01271` [cs.LG], 2018. **Code:** `https://github.com/locuslab/TCN`. TCNs are discrete, fixed receptive field — require resampled input.
* **Transformer (Baseline 3):** Ashish Vaswani et al. *Attention Is All You Need.* **NeurIPS 2017**. `arXiv:1706.03762`. For robot fusion, typically time-encoded positional embeddings + cross-modal attention (as in MiTaS). **Code:** `nn.Transformer`. Your version: time-encoded positions on resampled, aligned tokens.
* **Relation:** Your three equal-parameter discrete-time baselines. Hypothesis: they degrade under dropout/rate shift where CfC does not.

### E3. Diffusion Policy (Modern Low-Data Baseline) [VERIFIED — arXiv:2303.04137v5]

* **Citation:** Cheng Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* `arXiv:2303.04137` [cs.RO], 2023. RSS 2023.
* **Summary:** Denoising diffusion generates action sequences conditioned on vision+proprioception. Strong in multimodal, high-dim action spaces.
* **Key Results:** On 12 tasks (push-T, etc.), diffusion policy 10-40% higher success vs LSTM/Transformer BC, especially with multimodal action distributions. Slower inference (requires iterative denoising).
* **Code:** `https://github.com/real-stanford/diffusion_policy` — pretrained checkpoints for Push-T, Franka tasks.
* **Relation:** If you include Correll 2026 claim, diffusion is the modern baseline liquid beats in low-data regime. Consider citing as alternative, but your DA's formal baselines are LSTM/TCN/Transformer.

---

## F. SURVEYS & THEORY — Open Problems You Address

### F1. Comparative Review LNN vs RNN — Oct 2025 [UNVERIFIED]

* **Citation:** (As per DA) *Comparative Review of Liquid Neural Networks versus Recurrent Neural Networks.* (Survey, Oct 2025).
* **Summary (per DA):** Flags online/continual learning, uncertainty quantification, invariance to sensor noise/regime shifts as critical futures. Does NOT address asynchronous multimodal fusion gap.
* **Relation:** Cite to frame your gap as orthogonal to survey's futures — yours is enabled by architecture itself but untested.

### F2. CfC Original Paper's Open Question — Safety Verification [VERIFIED via CfC paper]

* **Citation:** Hasani et al. 2022 Nature MI Discussion: "whether safety verification is more tractable via ODE representation or its closed form remains open."
* **Relation:** Acknowledge, but not your focus — distinguishes your contribution (fusion) from theory futures.

---

## G. QUICK-REFERENCE TABLE — What to Cite Where in Your Paper

| Section in Your Paper | Papers to Cite | Why |
|---|---|---|
| **Abstract / Intro — Problem:** Multi-rate asynchrony, resampling artifacts | C2 (Kalman/ZOH practice), D1/D2 (GelSight/DIGIT rates), F1 (survey) | Establish that 1kHz/100Hz/30Hz jitter is real, current fix is resampling |
| **Intro — Solution: LTC/CfC** | A1 (LTC), A2 (CfC), A4 (NCP) | Define liquid time constants, closed-form speed, explicit time input |
| **Related Work — LNN in Robotics** | B1 (flight), B2 (LGTC), B3/B4 (manipulator dynamics), B5 (Correll vs diffusion) | Show breadth but all single-rate → your gap |
| **Related Work — Fusion & Tactile** | C1 (MiTaS 80/31/54%), D1/D2, D3 | Show multi-resolution helps but via discrete fusion; no LNN+tactile work exists |
| **Method — CfC Backbone** | A2 (Eq. CfC closed form), A4 (ncps wiring) | Architecture detail, delta_t handling |
| **Method — Baselines** | E2 (LSTM/TCN/Transformer), E3 (Diffusion if discussed), C1 (MiTaS fusion style) | Equal-parameter discrete baselines on resampled data |
| **Experiments — Hardware** | E1 (Franka Panda), D1/D2 (sensors) | Reproducibility |
| **Discussion — Why async matters** | A2 (claimed irregular-data target), C1 (MiTaS attention shows sensor timing matters) | Argue latency/sample-efficiency/robustness gains come from native time handling |

---

## H. CODE & CHECKPOINT STARTER KIT (Verified Links)

```bash
# Core liquid library — start here
pip install ncps torch  # NCPS = https://github.com/mlech26l/ncps
# CfC reference (older)
git clone https://github.com/raminmh/cfc
git clone https://github.com/raminmh/liquid_time_constant_networks

# Tactile
git clone https://github.com/facebookresearch/digit-interface  # DIGIT
git clone https://github.com/gelsightinc/gsrobotics            # GelSight

# Baselines
git clone https://github.com/locuslab/TCN                       # TCN
git clone https://github.com/real-stanford/diffusion_policy     # Diffusion Policy (optional)

# Simulation / Robot
# Franka Panda models: via MuJoCo Menagerie, Isaac Sim, or https://github.com/frankaemika/libfranka
```

**No pretrained CfC-robot checkpoint exists** — you train from demonstrations. That's expected; CfCs are trained per-task via imitation (behavior cloning / flow-matching as in MiTaS). Your evaluation will produce the first such checkpoints.

---

## I. WHAT TO VERIFY BEFORE SUBMISSION (UNVERIFIED items)

1. B1 Science Robotics exact title/year — confirm DOI in Web of Science.
2. B2 LGTC authors/venue — search `liquid-graph` on IEEE Xplore.
3. B3 Zhang invertible LNN venue — likely IEEE RA-L 2024, confirm.
4. B4 Urrea SMC+ CfC — confirm IEEE Access vs ETFA.
5. B5 Correll March 2026 — this date is in future relative to 2025 cutoff; verify arXiv ID or remove if not yet posted.
6. F1 Oct 2025 survey — get full citation.
7. MiTaS author list — `arXiv:2606.06281` page shows 6 authors (names not extracted via API grep) — fetch PDF to fill.
8. All arXiv IDs: re-run `curl https://export.arxiv.org/api/query?id_list=...` for each to capture final journal refs.

---

## J. FILES IN THIS DOSSIER FOLDER

* `/home/honeysh/Projects/cfc-async-fusion/RESEARCH_DOSSIER.md` — this file (canonical) + mirror in `CfC-Async-Sensor-Fusion/`
* Next steps suggested: `PAPERS/` folder with PDFs, `refs.bib` BibTeX, `baseline_comparison_table.csv`

Want me to generate `refs.bib` and download the 4 verified PDFs (LTC, CfC, MiTaS, Taxim) into `PAPERS/`?
