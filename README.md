# ZT-IPLS: Zero-Trust Architecture for AI-Driven IP Litigation Support

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research Code](https://img.shields.io/badge/status-research%20code-orange.svg)](#reproducibility-and-paper-alignment)

Official experiment repository for the manuscript:

> **Zero-Trust Architecture Patterns for Securing AI-Driven IP Litigation Support Systems**  
> Vineet Singh — University of Maryland, College Park  
> Manuscript under submission / revision

---

## What this repository contains

This repository contains the experimental code used to evaluate **ZT-IPLS**, a six-layer Zero Trust Architecture designed for AI-assisted IP litigation workflows.

The repository focuses on three evaluation questions:

1. Can a prompt-level policy enforcement point block adversarial prompt injection without breaking legitimate legal queries?
2. Can role and matter-aware access control prevent unauthorized access in a multi-party litigation workflow?
3. Can continuous monitoring reduce false positives while detecting insider-threat-like anomalous behavior?

The code is meant to support the paper's experimental section and to make the evaluation pipeline easier to inspect, rerun, and extend.

---

## ZT-IPLS in one page

ZT-IPLS extends Zero Trust ideas from network-centric access control into AI-assisted legal workflows, where the attack surface includes **prompts, LLM context windows, matter-scoped retrieval, and audit integrity**.

### Framework layers

| Layer | Purpose | Representative control | Main threats addressed |
|---|---|---|---|
| **L1 — Identity** | Role verification and authenticated session binding | role + matter scoping | T3 |
| **L2 — Prompt PEP** | Prompt-level policy enforcement before LLM routing | injection classifier / blocklist / guardrail | T1, T2, T3 |
| **L3 — LLM Engine** | Controlled model access and bounded inference | zero-retention routing | T2 |
| **L4 — Output Classification** | Output privilege / sensitivity labeling | document-class enforcement | T3 |
| **L5 — RAG Isolation** | Matter-aware segmentation for retrieval | per-matter namespace isolation | T4 |
| **L6 — Audit / Monitoring** | Telemetry-driven anomaly detection and evidence integrity | SIEM/CDM monitoring, append-only logs | T5 |

### Threat model used in the paper

| ID | Threat | Example consequence |
|---|---|---|
| **T1** | Prompt injection / context exfiltration | litigation strategy leakage |
| **T2** | LLM-mediated privilege breach | disclosure of protected work product |
| **T3** | Role escalation in a shared workflow | access beyond subject authorization |
| **T4** | Cross-matter retrieval contamination | conflict-of-interest / matter isolation failure |
| **T5** | Audit trail tampering | weakened evidentiary trust |

---

## Repository layout

```text
zt-ipls/
├── experiments/
│   ├── experiment1_prompt_injection.py
│   ├── experiment2_rbac.py
│   └── experiment3_anomaly_detection.py
├── docs/
│   └── framework_overview.md
├── figures/                 # created automatically when experiments run
├── results/                 # created automatically when experiments run
├── run_all_experiments.py   # convenience runner for the full evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Experiments

### Experiment 1 — Adversarial Prompt Injection Detection

**Goal:** validate the Layer 2 prompt-level Policy Enforcement Point.

**How it works:**
- generates a synthetic dataset of IP-litigation-style prompts
- trains and evaluates a fine-tuned **DistilBERT** classifier
- compares it against:
  - TF-IDF + Logistic Regression
  - a keyword blocklist baseline
  - repurposed zero-shot general-purpose guardrails

**Why it matters:** in a legal workflow, a useful safeguard must block adversarial prompts **without** over-blocking legitimate attorney or paralegal queries.

**Main outputs:**
- `results/experiment1_results.json`
- `figures/experiment1_confusion_matrix.png`
- `figures/experiment1_roc_curve.png`

**Important note:** DistilBERT fine-tuning is the only materially stochastic part of the repository. Small changes in prompt-detection metrics across runs are expected even with the same seed.

---

### Experiment 2 — Role-Based Access Control Enforcement

**Goal:** validate Layers 1 and 4 using a deterministic policy engine.

**How it works:**
- simulates five litigation workflow roles:
  - Attorney
  - Paralegal
  - IP Engineer
  - Client
  - Admin
- evaluates access decisions across three document classes:
  - Class A — Attorney Work Product
  - Class B — Internal Draft
  - Class C — Non-Privileged Output
- compares full ZT-IPLS enforcement against a weaker baseline without full matter-aware isolation

**Why it matters:** traditional role-only controls can miss cross-matter or workflow-context mistakes that are high risk in legal pipelines.

**Main outputs:**
- `results/experiment2_results.json`
- `figures/experiment2_access_matrix.png`
- `figures/experiment2_latency_distribution.png`

**Important note:** this experiment is rule-based and deterministic.

---

### Experiment 3 — Insider Threat Anomaly Detection

**Goal:** validate Layer 6 monitoring using network telemetry as a proxy for anomalous access behavior.

**How it works:**
- trains a **Random Forest + XGBoost** ensemble
- compares it against a Logistic Regression baseline
- uses **CICIDS-2017** when available
- falls back to a synthetic proxy dataset if the real dataset is unavailable

**Why it matters:** each false positive can trigger unnecessary compliance review or incident investigation. Lower false-positive rates make a monitoring layer more usable in practice.

**Main outputs:**
- `results/experiment3_results.json`
- `figures/experiment3_roc_comparison.png`
- `figures/experiment3_feature_importance.png`

**Important note:** for paper-quality results, use the real **CICIDS-2017** dataset rather than the synthetic fallback.

---

## Quick start

### 1) Clone and create an environment

```bash
git clone https://github.com/vineetsingh-vs/zt-ipls.git
cd zt-ipls

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run everything

```bash
python run_all_experiments.py
```

### 3) Or run individual experiments

```bash
python experiments/experiment2_rbac.py
python experiments/experiment1_prompt_injection.py
python experiments/experiment3_anomaly_detection.py
```

All figures are written to `figures/` and all JSON outputs are written to `results/`.

---

## Dataset note for Experiment 3

Experiment 3 uses the following CICIDS-2017 file when available:

- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

The script can automatically look in common local locations and also contains a download URL in the source. If the real dataset is unavailable, the experiment falls back to a synthetic proxy. That fallback is helpful for validating the code path, but it should **not** be used as the final paper result unless explicitly stated.

---

## Reproducibility and paper alignment

This repository is intended to support the paper, but there is an important distinction between **code reproducibility** and **paper-matched reproducibility**.

### What is stable

- **Experiment 2** is deterministic.
- **Experiment 3** is stable when run with the same dataset, seed, and package versions.
- Output locations and JSON structure are stable across runs.

### What can vary

- **Experiment 1** can vary slightly across runs because fine-tuning uses stochastic optimization.
- If Experiment 3 falls back to synthetic data, the results will not match the real-dataset run used for the paper.

### Best practice before linking this repository in a paper

Before adding this repository URL to a manuscript, it is a good idea to:

1. run the final paper-matched experiments once more
2. keep the exact JSON outputs in `results/`
3. keep the exact paper figures in `figures/`
4. tag the matching commit or create a release
5. mention in the paper which release / commit corresponds to the reported results

That makes the repository much stronger for reviewers because the reported tables, figures, and code all point to the same frozen artifact.

---

## Expected outputs

After a successful full run, the repository should contain files like these:

```text
results/
├── experiment1_results.json
├── experiment2_results.json
├── experiment3_results.json
└── consolidated_results.json

figures/
├── experiment1_confusion_matrix.png
├── experiment1_roc_curve.png
├── experiment2_access_matrix.png
├── experiment2_latency_distribution.png
├── experiment3_feature_importance.png
└── experiment3_roc_comparison.png
```

`results/consolidated_results.json` is intended as a convenient summary file for drafting or checking Section V of the manuscript.

---

## Limitations

- Experiment 1 uses a **synthetic** legal-prompt dataset rather than real privileged matter data.
- Experiment 2 is a controlled simulation of access policy enforcement, not a deployed enterprise policy engine.
- Experiment 3 uses network-intrusion telemetry as a practical proxy for anomalous access behavior rather than logs from a live legal AI platform.

These choices are deliberate: they preserve privacy and make the work reproducible, but they should be understood as experimental constraints.

---

## Safety and data handling

This repository is designed to avoid sensitive real-world legal material.

- prompts are synthetic
- access requests are simulated
- no client-confidential matter data is required
- no production credentials or private endpoints should be committed

If you adapt this repository for real workflows, use sanitized data and separate any private materials from the public research code.

---

## Citation

```bibtex
@article{singh2026ztipls,
  author  = {Singh, Vineet},
  title   = {Zero-Trust Architecture Patterns for Securing {AI}-Driven {IP} Litigation Support Systems},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Manuscript under submission / revision}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Suggested next cleanup before making the repo public

A good final polish would be to add:

- one small `sample_outputs/` folder or a tagged release with paper-matched results
- a short `CHANGELOG.md`
- one-line command examples in each experiment docstring that match the README exactly

That is optional, but it makes the repository look much more publication-ready.
