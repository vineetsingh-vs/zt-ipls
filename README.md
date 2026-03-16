# ZT-IPLS: Zero-Trust Architecture for AI-Driven IP Litigation Support

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research Code](https://img.shields.io/badge/status-research%20code-orange.svg)](#reproducibility-and-paper-alignment)

Official experiment repository for the manuscript:

> **Zero-Trust Architecture Patterns for Securing AI-Driven IP Litigation Support Systems**
> Vineet Singh — University of Maryland, College Park
> Submitted to IEEE Access, 2026

---

## What this repository contains

This repository contains the experimental code used to evaluate **ZT-IPLS**, a six-layer Zero Trust Architecture designed for AI-assisted IP litigation workflows.

The repository focuses on three evaluation questions:

1. Can a prompt-level policy enforcement point block adversarial prompt injection without breaking legitimate legal queries?
2. Can role and matter-aware access control prevent unauthorized access in a multi-party litigation workflow?
3. Can continuous monitoring reduce false positives while detecting insider-threat-like anomalous behavior?

---

## ZT-IPLS in one page

ZT-IPLS extends Zero Trust ideas from network-centric access control into AI-assisted legal workflows, where the attack surface includes **prompts, LLM context windows, matter-scoped retrieval, and audit integrity**.

### Framework layers

| Layer | Purpose | Representative control | Threats addressed |
|---|---|---|---|
| **L1 — Identity** | Role verification and authenticated session binding | role + matter scoping | T3 |
| **L2 — Prompt PEP** | Prompt-level policy enforcement before LLM routing | injection classifier / blocklist / guardrail | T1, T2, T3 |
| **L3 — LLM Engine** | Controlled model access and bounded inference | zero-retention routing | T2 |
| **L4 — Output Classification** | Output privilege / sensitivity labeling | document-class enforcement | T3 |
| **L5 — RAG Isolation** | Matter-aware segmentation for retrieval | per-matter namespace isolation | T4 |
| **L6 — Audit / Monitoring** | Telemetry-driven anomaly detection and evidence integrity | SIEM/CDM monitoring, append-only logs | T5 |

### Threat model

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
├── run_all_experiments.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Experiments

### Experiment 1 — Adversarial Prompt Injection Detection

**Goal:** validate the Layer 2 prompt-level Policy Enforcement Point.

**How it works:**
- generates a synthetic dataset of 200 IP-litigation-style prompts (100 normal, 100 adversarial)
- trains and evaluates a fine-tuned **DistilBERT** classifier
- compares it against TF-IDF + Logistic Regression, a keyword blocklist, and two zero-shot general-purpose guardrails (ToxicBERT, DistilBERT-SST2)

**Why it matters:** in a legal workflow, a useful safeguard must block adversarial prompts **without** over-blocking legitimate attorney or paralegal queries. The false positive rate is as important as the detection rate.

**Outputs:**
```
results/experiment1_results.json
figures/experiment1_confusion_matrix.png
figures/experiment1_roc_curve.png
```

**Note:** DistilBERT fine-tuning is stochastic. Small variation in detection metrics across runs (±2–5%) is expected even with the same seed.

---

### Experiment 2 — Role-Based Access Control Enforcement

**Goal:** validate Layers 1 and 4 using a deterministic policy engine.

**How it works:**
- simulates 150 access requests across five roles (Attorney, Paralegal, IP Engineer, Client, Admin) and three document classes (Class A Work Product, Class B Internal Draft, Class C Non-Privileged)
- compares ZT-IPLS full enforcement (role + matter isolation) against a baseline with role-only RBAC and no matter-aware isolation

**Why it matters:** role-only controls miss cross-matter access attempts. Matter-scope binding is necessary to prevent T4 contamination in shared litigation pipelines.

**Outputs:**
```
results/experiment2_results.json
figures/experiment2_access_matrix.png
figures/experiment2_latency_distribution.png
```

**Note:** fully deterministic — identical results every run.

---

### Experiment 3 — Insider Threat Anomaly Detection

**Goal:** validate Layer 6 monitoring using network telemetry as a proxy for anomalous access behavior.

**How it works:**
- trains a **Random Forest + XGBoost** soft-voting ensemble
- compares against a Logistic Regression baseline
- uses **CICIDS-2017** (real dataset) when available, synthetic proxy as fallback

**Why it matters:** each false positive can trigger unnecessary compliance review. A lower false positive rate makes the monitoring layer operationally viable in legal workflows.

**Outputs:**
```
results/experiment3_results.json
figures/experiment3_roc_comparison.png
figures/experiment3_feature_importance.png
```

**Note:** use the real CICIDS-2017 dataset for paper-quality results. The synthetic fallback is only suitable for code validation.

---

## Quick start

### 1. Clone and set up environment

```bash
git clone https://github.com/vineetsingh-vs/zt-ipls.git
cd zt-ipls

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS Intel note:** if torch fails to install:
```bash
pip install "torch>=2.2.0" "numpy<2"
brew install libomp
pip install "transformers==4.38.2"
pip install scikit-learn xgboost pandas matplotlib tqdm requests
```

### 2. Run everything

```bash
python run_all_experiments.py
```

### 3. Or run individually

```bash
python experiments/experiment2_rbac.py    # 30 seconds — good first test
python experiments/experiment1_prompt_injection.py   # ~15 min
python experiments/experiment3_anomaly_detection.py  # ~40 min
```

All figures are written to `figures/` and all JSON outputs are written to `results/` using absolute paths — results land in the right place regardless of which directory you run from.

---

## Dataset note for Experiment 3

Experiment 3 uses `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` from CICIDS-2017.

Download from: https://www.unb.ca/cic/datasets/ids-2017.html

Place the file in the project root. The script searches the project root, `~/Downloads`, and `~/Desktop` automatically. If the file is not found, the experiment falls back to a synthetic proxy — useful for testing the code path but not for reporting results.

---

## Reproducibility and paper alignment

### What is stable

- **Experiment 2** is deterministic — identical results every run
- **Experiment 3** is stable with the same dataset, seed, and package versions
- Output file locations and JSON structure are stable across runs

### What can vary

- **Experiment 1** varies slightly across runs due to stochastic fine-tuning
- **Experiment 3** produces different results on synthetic vs real data

### Before linking this repository in a paper

1. Run the final paper-matched experiments once more
2. Keep the exact JSON outputs in `results/`
3. Keep the exact figures in `figures/`
4. Tag the matching commit or create a release
5. Reference the specific tag or commit hash in the manuscript

---

## Expected outputs after a full run

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

---

## Limitations

- Experiment 1 uses a synthetic legal-prompt dataset rather than real privileged matter data
- Experiment 2 is a controlled simulation of access policy enforcement, not a deployed enterprise system
- Experiment 3 uses network-intrusion telemetry as a proxy for anomalous access behavior rather than logs from a live legal AI platform

These choices are deliberate — they preserve privacy and make the work reproducible, but should be understood as experimental constraints.

---

## Safety and data handling

This repository contains no sensitive real-world legal material.

- Prompts are synthetic
- Access requests are simulated
- No client-confidential matter data is required
- No production credentials or private endpoints are used

If you adapt this repository for real workflows, use sanitized data and keep any private materials out of the public research code.

---

## Citation

```bibtex
@article{singh2026ztipls,
  author  = {Singh, Vineet},
  title   = {Zero-Trust Architecture Patterns for Securing {AI}-Driven {IP} Litigation Support Systems},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

Released under the [MIT License](LICENSE).