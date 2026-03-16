"""
=============================================================================
ZT-IPLS EXPERIMENT 2: Role-Based Access Control Enforcement
=============================================================================
Paper: "Zero-Trust Architecture Patterns for Securing AI-Driven
        IP Litigation Support Systems"

What this experiment does
--------------------------
Simulates the Layer 1 (Identity/Role Verification) and Layer 4 (Output
Classification) components of ZT-IPLS enforcing RBAC across a five-role
IP litigation pipeline. 150 access requests are generated across three
privilege levels and five subject roles. The ZTA policy engine is tested
for correctness, latency, and insider threat containment.

No ML model required — this is a deterministic policy engine test.
Runtime: < 60 seconds.

Roles and privilege levels
--------------------------
  Role 1 — Attorney:      Level 3 (full work product access)
  Role 2 — Paralegal:     Level 2 (internal draft access)
  Role 3 — IP Engineer:   Level 2 (technical document access)
  Role 4 — Client:        Level 1 (non-privileged output only)
  Role 5 — Admin:         Level 1 (system access, no work product)

Document classes
----------------
  Class A — Attorney Work Product:  requires Level 3
  Class B — Internal Draft:         requires Level 2
  Class C — Non-Privileged Output:  requires Level 1

Install dependencies
--------------------
    pip install numpy pandas matplotlib scikit-learn

Run
---
    python experiment2_rbac.py

Output
------
  - Console: full access control results table
  - File:    experiment2_results.json
  - File:    experiment2_access_matrix.png
  - File:    experiment2_latency_distribution.png

=============================================================================
"""

import json, random, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings('ignore')

SEED = 42

# ── Output paths (always relative to project root, regardless of cwd) ─────────
import pathlib as _pathlib
_ROOT    = _pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"
_FIGURES = _ROOT / "figures"
_RESULTS.mkdir(parents=True, exist_ok=True)
_FIGURES.mkdir(parents=True, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 1. ROLE AND DOCUMENT DEFINITIONS
# =============================================================================

# Five-role hierarchy mapped to privilege levels
ROLES = {
    'Attorney':    {'level': 3, 'description': 'Full work product access'},
    'Paralegal':   {'level': 2, 'description': 'Internal draft access'},
    'IP_Engineer': {'level': 2, 'description': 'Technical document access'},
    'Client':      {'level': 1, 'description': 'Non-privileged output only'},
    'Admin':       {'level': 1, 'description': 'System access, no work product'},
}

# Three document classes mapped to minimum required privilege levels
DOCUMENT_CLASSES = {
    'Class_A_Attorney_Work_Product': {
        'required_level': 3,
        'description':    'Attorney mental impressions, strategy memos, claim construction positions',
        'privilege':      'Attorney Work Product (FRCP 26(b)(3))',
    },
    'Class_B_Internal_Draft': {
        'required_level': 2,
        'description':    'Internal draft briefs, paralegal summaries, technical analysis',
        'privilege':      'Internal Confidential',
    },
    'Class_C_Non_Privileged': {
        'required_level': 1,
        'description':    'Public filings, docket entries, published prior art',
        'privilege':      'Non-Privileged',
    },
}

MATTER_NUMBERS = [
    'US-2024-0341', 'US-2024-0892', 'US-2023-1147',
    'US-2024-0558', 'US-2025-0012', 'US-2023-0776',
]

# Specific document types within each class
DOCUMENT_TYPES = {
    'Class_A_Attorney_Work_Product': [
        'Claim construction strategy memo',
        'Invalidity contention analysis',
        'Settlement position paper',
        'Litigation strategy assessment',
        'Expert witness preparation notes',
        'Deposition strategy outline',
        'IPR petition draft — attorney section',
        'Licensing negotiation position memo',
    ],
    'Class_B_Internal_Draft': [
        'Draft IPR petition — technical sections',
        'Prior art claim chart draft',
        'Prosecution history summary',
        'Technical declaration draft',
        'Discovery request draft',
        'Deposition outline — technical topics',
        'Internal case status summary',
        'Prior art search report',
    ],
    'Class_C_Non_Privileged': [
        'Published USPTO filing',
        'Prior art patent document',
        'PTAB public docket entry',
        'Court public filing',
        'Published claim chart',
        'Prior art search results (public)',
        'Docket deadline calendar',
        'Public case status summary',
    ],
}


# =============================================================================
# 2. ZTA POLICY ENGINE (Layer 1 + Layer 4 enforcement)
# =============================================================================

class ZTAPolicyEngine:
    """
    Implements the NIST SP 800-207 Policy Engine (PE) and
    Policy Administrator (PA) for the IP litigation ZTA pipeline.

    Decision logic:
      PERMIT  — subject.privilege_level >= document.required_level
                AND subject.matter_id in document.authorized_matters
                AND session is authenticated
      DENY    — any condition fails
    """

    def __init__(self, enforce_matter_isolation: bool = True,
                 log_all_decisions: bool = True):
        self.enforce_matter_isolation = enforce_matter_isolation
        self.log_all_decisions = log_all_decisions
        self.audit_log = []
        self._decision_count = 0

    def evaluate(self, subject_role: str, subject_matter_scope: list,
                 document_class: str, document_matter: str,
                 session_authenticated: bool,
                 is_insider_attack: bool = False) -> dict:
        """
        Evaluate a single access request against ZTA policy.

        Parameters
        ----------
        subject_role          : Role of the requesting subject
        subject_matter_scope  : List of matter IDs the subject is authorized for
        document_class        : Document classification (A/B/C)
        document_matter       : Matter ID the document belongs to
        session_authenticated : Whether the session passed L1 MFA
        is_insider_attack     : Flag for simulation — marks requests where
                                an insider attempts unauthorized access

        Returns
        -------
        dict with decision, reason, latency_ms
        """
        t0 = time.perf_counter()
        self._decision_count += 1

        role_info = ROLES[subject_role]
        doc_info  = DOCUMENT_CLASSES[document_class]

        subject_level  = role_info['level']
        required_level = doc_info['required_level']

        # ── Policy checks (in order of NIST ZTA tenet priority) ──────────────

        # Check 1: Session authentication (L1 MFA gate)
        if not session_authenticated:
            decision = 'DENY'
            reason   = 'Unauthenticated session — L1 MFA failed'
            policy_rule = 'ZTA-TENET-1: All communication secured'

        # Check 2: Privilege level sufficiency
        elif subject_level < required_level:
            decision = 'DENY'
            reason   = (f'Insufficient privilege: role {subject_role} '
                        f'(level {subject_level}) requires level {required_level} '
                        f'for {document_class}')
            policy_rule = 'ZTA-TENET-2: Least-privilege access enforcement'

        # Check 3: Matter scope isolation (L5 namespace enforcement)
        elif (self.enforce_matter_isolation
              and document_matter not in subject_matter_scope):
            decision = 'DENY'
            reason   = (f'Matter scope violation: subject not authorized '
                        f'for matter {document_matter}')
            policy_rule = 'ZTA-TENET-3: Micro-segmentation (T4 prevention)'

        # Check 4: All checks passed
        else:
            decision = 'PERMIT'
            reason   = (f'Access granted: level {subject_level} >= '
                        f'required {required_level}, matter in scope')
            policy_rule = 'ZTA-TENET-ALL: Policy satisfied'

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Audit log entry (L6) ──────────────────────────────────────────────
        log_entry = {
            'request_id':       self._decision_count,
            'subject_role':     subject_role,
            'subject_level':    subject_level,
            'document_class':   document_class,
            'required_level':   required_level,
            'document_matter':  document_matter,
            'matter_in_scope':  document_matter in subject_matter_scope,
            'authenticated':    session_authenticated,
            'decision':         decision,
            'reason':           reason,
            'policy_rule':      policy_rule,
            'is_insider_attack': is_insider_attack,
            'latency_ms':       round(latency_ms, 4),
        }
        self.audit_log.append(log_entry)

        return {
            'decision':    decision,
            'reason':      reason,
            'policy_rule': policy_rule,
            'latency_ms':  latency_ms,
        }

    def get_audit_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.audit_log)


# =============================================================================
# 3. REQUEST DATASET GENERATION
# =============================================================================

def generate_access_requests(n: int = 150, seed: int = SEED) -> pd.DataFrame:
    """
    Generate 150 access requests covering:
      - Legitimate requests (authorized access)
      - Unauthorized requests (role insufficient for document class)
      - Insider attack simulation (authorized role, wrong matter)
      - Unauthenticated session attempts
    """
    random.seed(seed)
    requests = []

    roles      = list(ROLES.keys())
    doc_classes = list(DOCUMENT_CLASSES.keys())

    # ── 90 legitimate authorized requests (60% of dataset) ───────────────────
    # These should all be PERMIT
    for _ in range(90):
        role     = random.choice(roles)
        level    = ROLES[role]['level']
        # Pick a document class the role CAN access
        eligible = [dc for dc, info in DOCUMENT_CLASSES.items()
                    if info['required_level'] <= level]
        doc_class = random.choice(eligible)
        matter   = random.choice(MATTER_NUMBERS)
        requests.append({
            'subject_role':           role,
            'subject_matter_scope':   [matter],   # authorized for this matter
            'document_class':         doc_class,
            'document_matter':        matter,
            'session_authenticated':  True,
            'is_insider_attack':      False,
            'expected_decision':      'PERMIT',
            'request_type':           'Legitimate',
        })

    # ── 30 unauthorized role-level attempts (20% of dataset) ─────────────────
    # T3: subject tries to access document above their privilege level
    for _ in range(30):
        # Force a role that cannot access Class A
        role = random.choice(['Paralegal', 'IP_Engineer', 'Client', 'Admin'])
        doc_class = 'Class_A_Attorney_Work_Product'   # requires level 3
        matter   = random.choice(MATTER_NUMBERS)
        requests.append({
            'subject_role':           role,
            'subject_matter_scope':   [matter],
            'document_class':         doc_class,
            'document_matter':        matter,
            'session_authenticated':  True,
            'is_insider_attack':      True,    # T3 role escalation simulation
            'expected_decision':      'DENY',
            'request_type':           'T3_Role_Escalation',
        })

    # ── 15 cross-matter access attempts (10% of dataset) ─────────────────────
    # T4: subject authorized for Matter A tries to access Matter B document
    for _ in range(15):
        role  = random.choice(['Attorney', 'Paralegal', 'IP_Engineer'])
        level = ROLES[role]['level']
        eligible  = [dc for dc, info in DOCUMENT_CLASSES.items()
                     if info['required_level'] <= level]
        doc_class = random.choice(eligible)
        # Give subject scope for one matter, document is from a different one
        authorized_matter = MATTER_NUMBERS[0]
        doc_matter        = MATTER_NUMBERS[1]   # different matter — should DENY
        requests.append({
            'subject_role':           role,
            'subject_matter_scope':   [authorized_matter],
            'document_class':         doc_class,
            'document_matter':        doc_matter,
            'session_authenticated':  True,
            'is_insider_attack':      True,    # T4 cross-matter simulation
            'expected_decision':      'DENY',
            'request_type':           'T4_CrossMatter',
        })

    # ── 15 unauthenticated session attempts (10% of dataset) ─────────────────
    for _ in range(15):
        role      = random.choice(roles)
        doc_class = random.choice(doc_classes)
        matter    = random.choice(MATTER_NUMBERS)
        requests.append({
            'subject_role':           role,
            'subject_matter_scope':   [matter],
            'document_class':         doc_class,
            'document_matter':        matter,
            'session_authenticated':  False,   # unauthenticated
            'is_insider_attack':      False,
            'expected_decision':      'DENY',
            'request_type':           'Unauthenticated',
        })

    random.shuffle(requests)
    return pd.DataFrame(requests)


# =============================================================================
# 4. BASELINE: No-ZTA flat access control (role-only, no matter isolation)
# =============================================================================

def baseline_no_zta(subject_role: str, document_class: str) -> str:
    """
    Simulates a pre-ZTA access control system:
    - Checks role level vs document level (basic RBAC)
    - Does NOT enforce matter isolation (T4 vulnerability)
    - Does NOT check session authentication robustly
    Returns PERMIT or DENY.
    """
    level    = ROLES[subject_role]['level']
    required = DOCUMENT_CLASSES[document_class]['required_level']
    return 'PERMIT' if level >= required else 'DENY'


# =============================================================================
# 5. METRICS
# =============================================================================

def compute_rbac_metrics(df: pd.DataFrame, decision_col: str,
                         expected_col: str = 'expected_decision',
                         latency_col: str  = None) -> dict:
    """Compute access control accuracy and security metrics."""
    correct = (df[decision_col] == df[expected_col]).sum()
    total   = len(df)
    acc     = correct / total

    # Unauthorized access incidents = cases that should DENY but got PERMIT
    should_deny = df[df[expected_col] == 'DENY']
    unauthorized_granted = (should_deny[decision_col] == 'PERMIT').sum()

    # Legitimate requests incorrectly blocked
    should_permit  = df[df[expected_col] == 'PERMIT']
    legitimate_blocked = (should_permit[decision_col] == 'DENY').sum()

    metrics = {
        'total_requests':            int(total),
        'correct_decisions':         int(correct),
        'accuracy':                  round(acc, 4),
        'unauthorized_access_count': int(unauthorized_granted),
        'legitimate_blocked_count':  int(legitimate_blocked),
    }
    if latency_col and latency_col in df.columns:
        metrics['mean_latency_ms'] = round(float(df[latency_col].mean()), 4)
        metrics['p95_latency_ms']  = round(float(df[latency_col].quantile(0.95)), 4)
        metrics['max_latency_ms']  = round(float(df[latency_col].max()), 4)

    return metrics


# =============================================================================
# 6. PLOTTING
# =============================================================================

def plot_access_matrix(df_results: pd.DataFrame, output_path: str):
    """
    Heatmap of decision outcomes by role and document class.
    Shows the enforcement pattern of the ZTA policy engine.
    """
    roles_ordered = ['Attorney', 'Paralegal', 'IP_Engineer', 'Client', 'Admin']
    doc_classes_short = {
        'Class_A_Attorney_Work_Product': 'Class A\n(Atty Work Product)',
        'Class_B_Internal_Draft':        'Class B\n(Internal Draft)',
        'Class_C_Non_Privileged':        'Class C\n(Non-Privileged)',
    }

    # Build permit rate matrix
    matrix = np.zeros((len(roles_ordered), len(doc_classes_short)))
    for i, role in enumerate(roles_ordered):
        for j, dc in enumerate(doc_classes_short.keys()):
            subset = df_results[
                (df_results['subject_role'] == role) &
                (df_results['document_class'] == dc)
            ]
            if len(subset) > 0:
                matrix[i, j] = (subset['zta_decision'] == 'PERMIT').mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='PERMIT rate')

    ax.set_xticks(range(len(doc_classes_short)))
    ax.set_yticks(range(len(roles_ordered)))
    ax.set_xticklabels(list(doc_classes_short.values()), fontsize=9)
    ax.set_yticklabels(roles_ordered, fontsize=9)
    ax.set_xlabel('Document Class')
    ax.set_ylabel('Subject Role')
    ax.set_title('ZT-IPLS Access Control Matrix\n'
                 'PERMIT rate by role and document class', fontsize=10)

    for i in range(len(roles_ordered)):
        for j in range(len(doc_classes_short)):
            val = matrix[i, j]
            color = 'white' if val < 0.5 or val > 0.85 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_latency_distribution(df_results: pd.DataFrame, output_path: str):
    """Histogram of ZTA policy decision latency."""
    latencies = df_results['latency_ms'].values * 1000  # convert to microseconds
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Distribution
    axes[0].hist(latencies, bins=30, color='#2563eb', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
    axes[0].axvline(np.mean(latencies), color='red', linestyle='--',
                    lw=1.2, label=f'Mean: {np.mean(latencies):.1f} μs')
    axes[0].axvline(np.percentile(latencies, 95), color='orange',
                    linestyle=':', lw=1.2,
                    label=f'P95: {np.percentile(latencies, 95):.1f} μs')
    axes[0].set_xlabel('Policy Decision Latency (μs)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Policy Engine Latency Distribution', fontsize=10)
    axes[0].legend(fontsize=8)

    # Per-request type
    request_types = df_results['request_type'].unique()
    type_latencies = [
        df_results[df_results['request_type'] == rt]['latency_ms'].values * 1000
        for rt in request_types
    ]
    axes[1].boxplot(type_latencies, labels=request_types, vert=True)
    axes[1].set_xlabel('Request Type')
    axes[1].set_ylabel('Latency (μs)')
    axes[1].set_title('Latency by Request Type', fontsize=10)
    axes[1].tick_params(axis='x', rotation=20, labelsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("  ZT-IPLS Experiment 2: Role-Based Access Control Enforcement")
    print("  Paper: Zero-Trust Architecture for AI-Driven IP Litigation Support")
    print("=" * 78)

    # ── Generate requests ─────────────────────────────────────────────────────
    print("\n[1/5] Generating 150 access request simulation dataset...")
    df = generate_access_requests(n=150, seed=SEED)

    type_counts = df['request_type'].value_counts()
    for rtype, count in type_counts.items():
        print(f"  {rtype:<30} {count:>3} requests")
    print(f"  Total: {len(df)}")
    print(f"  Expected PERMIT: {(df['expected_decision']=='PERMIT').sum()}")
    print(f"  Expected DENY:   {(df['expected_decision']=='DENY').sum()}")

    # ── ZTA policy engine ─────────────────────────────────────────────────────
    print("\n[2/5] Running ZT-IPLS policy engine (L1 + L4 enforcement)...")
    engine = ZTAPolicyEngine(enforce_matter_isolation=True)
    zta_decisions = []
    zta_latencies = []

    for _, row in df.iterrows():
        result = engine.evaluate(
            subject_role          = row['subject_role'],
            subject_matter_scope  = row['subject_matter_scope'],
            document_class        = row['document_class'],
            document_matter       = row['document_matter'],
            session_authenticated = row['session_authenticated'],
            is_insider_attack     = row['is_insider_attack'],
        )
        zta_decisions.append(result['decision'])
        zta_latencies.append(result['latency_ms'])

    df['zta_decision'] = zta_decisions
    df['latency_ms']   = zta_latencies

    # ── Baseline: no-ZTA ─────────────────────────────────────────────────────
    print("[3/5] Running no-ZTA baseline (role-only, no matter isolation)...")
    baseline_decisions = [
        baseline_no_zta(row['subject_role'], row['document_class'])
        for _, row in df.iterrows()
    ]
    df['baseline_decision'] = baseline_decisions

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[4/5] Computing metrics...")
    zta_metrics      = compute_rbac_metrics(df, 'zta_decision',
                                            latency_col='latency_ms')
    baseline_metrics = compute_rbac_metrics(df, 'baseline_decision')

    print("\n" + "=" * 78)
    print("  EXPERIMENT 2 RESULTS — Role-Based Access Control Enforcement")
    print("=" * 78)
    print(f"\n  {'Metric':<45} {'ZT-IPLS':>12} {'No-ZTA':>12}")
    print("  " + "-" * 70)
    metrics_to_show = [
        ('Total Requests',               'total_requests'),
        ('Correct Decisions',            'correct_decisions'),
        ('Access Control Accuracy',      'accuracy'),
        ('Unauthorized Access Incidents','unauthorized_access_count'),
        ('Legitimate Requests Blocked',  'legitimate_blocked_count'),
    ]
    for label, key in metrics_to_show:
        zval = zta_metrics.get(key, 'N/A')
        bval = baseline_metrics.get(key, 'N/A')
        if isinstance(zval, float):
            print(f"  {label:<45} {zval:>12.4f} {bval:>12.4f}")
        else:
            print(f"  {label:<45} {zval:>12} {bval:>12}")
    print(f"\n  {'Mean Policy Decision Latency (ms)':<45} "
          f"{zta_metrics.get('mean_latency_ms', 'N/A'):>12}")
    print(f"  {'P95 Policy Decision Latency (ms)':<45} "
          f"{zta_metrics.get('p95_latency_ms', 'N/A'):>12}")
    print("=" * 78)

    # ── Per-attack-type breakdown ─────────────────────────────────────────────
    print("\n  Per-request-type breakdown (ZT-IPLS):")
    print(f"  {'Request Type':<30} {'Total':>6} {'Correct':>8} {'Acc':>8}")
    print("  " + "-" * 55)
    for rtype in df['request_type'].unique():
        subset = df[df['request_type'] == rtype]
        correct = (subset['zta_decision'] == subset['expected_decision']).sum()
        acc = correct / len(subset)
        print(f"  {rtype:<30} {len(subset):>6} {correct:>8} {acc:>8.4f}")
    print()

    # ── Audit log summary ─────────────────────────────────────────────────────
    audit_df = engine.get_audit_log_df()
    print(f"  Audit log entries generated: {len(audit_df)}")
    print(f"  Total PERMIT decisions:      "
          f"{(audit_df['decision']=='PERMIT').sum()}")
    print(f"  Total DENY decisions:        "
          f"{(audit_df['decision']=='DENY').sum()}")
    insider_attacks = audit_df[audit_df['is_insider_attack'] == True]
    insider_blocked = (insider_attacks['decision'] == 'DENY').sum()
    print(f"  Insider attack attempts:     {len(insider_attacks)}")
    print(f"  Insider attacks blocked:     {insider_blocked} "
          f"({insider_blocked/max(len(insider_attacks),1)*100:.1f}%)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_access_matrix(df, str(_FIGURES / 'experiment2_access_matrix.png'))
    plot_latency_distribution(df, str(_FIGURES / 'experiment2_latency_distribution.png'))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    print("\n[5/5] Saving results to experiment2_results.json...")
    insider_total   = len(insider_attacks)
    insider_blocked_count = int(insider_blocked)

    output = {
        'experiment': 'Experiment 2 — Role-Based Access Control Enforcement',
        'paper':      'ZT-IPLS: Zero-Trust Architecture for AI-Driven IP Litigation',
        'dataset': {
            'total_requests':    len(df),
            'legitimate':        int((df['request_type']=='Legitimate').sum()),
            't3_role_escalation': int((df['request_type']=='T3_Role_Escalation').sum()),
            't4_cross_matter':    int((df['request_type']=='T4_CrossMatter').sum()),
            'unauthenticated':    int((df['request_type']=='Unauthenticated').sum()),
            'random_seed':        SEED,
        },
        'zta_metrics':      zta_metrics,
        'baseline_metrics': baseline_metrics,
        'insider_threat': {
            'total_insider_attempts': int(insider_total),
            'blocked_by_zta':         insider_blocked_count,
            'block_rate':             round(insider_blocked_count / max(insider_total, 1), 4),
        },
        'section_v_numbers': {
            'NOTE': 'Use these numbers in Section V of the paper',
            'access_control_accuracy':       zta_metrics['accuracy'],
            'unauthorized_access_incidents': zta_metrics['unauthorized_access_count'],
            'legitimate_blocked':            zta_metrics['legitimate_blocked_count'],
            'insider_block_rate':            round(insider_blocked_count / max(insider_total, 1), 4),
            'mean_policy_latency_ms':        zta_metrics.get('mean_latency_ms'),
            'p95_policy_latency_ms':         zta_metrics.get('p95_latency_ms'),
            'baseline_unauthorized_access':  baseline_metrics['unauthorized_access_count'],
        },
    }

    with open(str(_RESULTS / 'experiment2_results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  DONE. Copy these numbers into Section V of your paper:")
    print("=" * 78)
    print(f"  Access Control Accuracy:      {zta_metrics['accuracy']:.4f} "
          f"({zta_metrics['accuracy']*100:.1f}%)")
    print(f"  Unauthorized Access (ZTA):    {zta_metrics['unauthorized_access_count']}")
    print(f"  Unauthorized Access (no ZTA): {baseline_metrics['unauthorized_access_count']}")
    print(f"  Insider Attacks Blocked:      {insider_blocked_count} / {insider_total} "
          f"({insider_blocked_count/max(insider_total,1)*100:.1f}%)")
    print(f"  Mean Policy Latency:          "
          f"{zta_metrics.get('mean_latency_ms', 'N/A')} ms")
    print(f"  P95 Policy Latency:           "
          f"{zta_metrics.get('p95_latency_ms', 'N/A')} ms")
    print("=" * 78)
    print("\n  Output files:")
    print(f"    {str(_RESULTS / 'experiment2_results.json'):<60} ← JSON results")
    print(f"    {str(_FIGURES / 'experiment2_access_matrix.png'):<60} ← Figure 3")
    print(f"    {str(_FIGURES / 'experiment2_latency_distribution.png'):<60} ← Figure 4")
    print()


if __name__ == '__main__':
    main()
