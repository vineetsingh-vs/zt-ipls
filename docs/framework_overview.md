# ZT-IPLS Framework Architecture Reference

## Overview

ZT-IPLS (Zero-Trust IP Litigation Support) is a six-layer security framework
that enforces the NIST SP 800-207 "never trust, always verify" principle across
every stage of an AI-driven IP litigation support pipeline.

---

## System Model

The pipeline under protection comprises:

1. **Subject population** — Attorneys (L3), Paralegals (L2), IP Engineers (L2),
   Clients (L1), Admins (L1)
2. **RAG subsystem** — Retrieves case law, prior art, internal litigation
   documents from one or more vector stores
3. **LLM inference endpoints** — Receives prompts, generates invalidity
   contentions, IPR petitions, claim charts, litigation strategy
4. **Document management system (DMS)** — Stores, versions, governs access
   to generated work product

---

## Layer Reference

### L1 — Identity and Role Verification
**NIST Component:** Policy Engine (PE) + Subject Database

- Multi-factor authentication (MFA) per session
- Role hierarchy binding: Attorney (3) > Paralegal (2) > IP Engineer (2) > Client (1) > Admin (1)
- Device posture evaluation
- Matter-scope binding at session establishment
- Trust score propagation to downstream layers

**Threats mitigated:** T3 (partial — prevents escalation at session level)

---

### L2 — Prompt-Level Policy Enforcement Point *(Novel Contribution)*
**NIST Component:** Policy Enforcement Point (PEP)

The first prompt-level PEP in ZTA literature. Three internal stages:

**Stage 1 — Privilege scope validation**
- Extracts matter identifier and intended action from incoming prompt
- Verifies subject's authenticated role authorizes the requested scope
- Blocks before LLM inference if unauthorized

**Stage 2 — NLP injection classifier**
- Fine-tuned transformer classifier (DistilBERT)
- Detects direct injection (override instructions, context extraction)
- Detects indirect injection (embedded in retrieved documents)
- Detects role escalation prompts
- Blocks and logs prompts exceeding risk threshold

**Stage 3 — Context window endpoint guard**
- Verifies prompt routes only to approved zero-retention LLM endpoints
- Blocks routing to unapproved third-party endpoints
- Enforces T2 mitigation before context window transmission

**Mean latency:** ~47ms end-to-end (see Experiment 1 results)
**Threats mitigated:** T1, T2, T3

---

### L3 — LLM Inference Engine
**NIST Component:** Enterprise Resource (protected)

- Receives only prompts that passed L2 PEP gate
- Operates under enforced zero-data-retention agreements
- Framework-agnostic: GPT-4o, Claude, Llama 3, or fine-tuned legal models
- Outputs immediately captured by L4 before delivery

**Threats mitigated:** T2 (context window leakage)

---

### L4 — Output Classification and Privilege Tagging
**NIST Component:** Policy Administrator (PA)

- BERT-based work product classifier on all LLM outputs
- Three privilege levels:
  - **Class A** — Attorney Work Product (Level 3 subjects only)
  - **Class B** — Internal Draft (Levels 2–3)
  - **Class C** — Non-Privileged Output (all authorized subjects)
- Cryptographic privilege tag bound to document object
- Tag enforced by DMS on all subsequent access requests

**Threats mitigated:** T3 (post-generation access control)

---

### L5 — Cross-Matter Isolation and RAG Namespace Enforcement
**NIST Component:** Micro-segmentation (data layer)

- Per-matter embedding namespace in vector store
- Metadata filter enforcement at retrieval API level
- Namespace established at L1 session binding
- Prevents cross-matter embedding proximity attacks
- Compatible with Pinecone, Weaviate, pgvector

**Threats mitigated:** T4 (cross-matter RAG contamination)

---

### L6 — Immutable Audit and Compliance Log
**NIST Component:** CDM + SIEM

- Append-only log of all pipeline interactions
- Fields: subject identity, role, matter scope, prompt hash, classifier
  decision, LLM endpoint, output privilege class, document access events
- SHA-256 hash-chained entries — retrospective modification detectable
- Satisfies ABA Model Rule 5.3 supervision requirements
- Satisfies eDiscovery obligations (chain of custody)

**Threats mitigated:** T5 (audit trail tampering)

---

## Compliance Mapping

| Requirement | Framework Component | Layer |
|-------------|--------------------|----|
| ABA Rule 1.6 (confidentiality) | Role verification + prompt PEP | L1, L2 |
| ABA Rule 1.7 (conflict of interest) | Per-matter namespace isolation | L5 |
| ABA Rule 5.3 (supervision) | Immutable audit log | L6 |
| FRCP 26(b)(3) (work product) | Zero-retention endpoint routing | L2, L3 |
| DTSA §1839 (trade secrets) | Output classification + privilege tag | L4 |
| ABA Formal Opinion 512 | Full framework | L1–L6 |

---

## NIST SP 800-207 Tenet Mapping

| NIST Tenet | ZT-IPLS Implementation |
|------------|------------------------|
| All data sources are resources | LLM endpoints treated as untrusted resources |
| All communication secured | Zero-retention, approved-endpoint enforcement (L2, L3) |
| Per-session resource access | Matter-scoped session binding (L1) |
| Access determined by dynamic policy | Trust score + classifier decision (L1, L2) |
| Enterprise monitors all assets | Hash-chained audit log (L6) |
| Never trust, always verify | Prompt-level PEP gate (L2) |
| Least-privilege access | Role hierarchy + privilege tagging (L1, L4) |
