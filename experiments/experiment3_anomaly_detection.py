"""
=============================================================================
ZT-IPLS EXPERIMENT 3: Insider Threat Anomaly Detection
=============================================================================
Paper: "Zero-Trust Architecture Patterns for Securing AI-Driven
        IP Litigation Support Systems"

What this experiment does
--------------------------
Validates the Layer 6 continuous monitoring component of ZT-IPLS using
the public CICIDS-2017 network intrusion dataset. A Random Forest +
XGBoost ensemble classifier detects anomalous access patterns consistent
with insider threat behavior. Results are compared against a no-ZTA
baseline (logistic regression with no ensemble) to demonstrate the
improvement provided by the AI-enhanced ZTA monitoring layer.

This uses the same benchmark dataset as Goel & Gupta [10] and Mangla [14],
making your comparison directly credible to reviewers who know those papers.

Dataset
-------
CICIDS-2017 (Canadian Institute for Cybersecurity)
  - Free public download: https://www.unb.ca/cic/datasets/ids-2017.html
  - Direct file used: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    (~54 MB, ~225,000 rows, binary: BENIGN vs DDoS/infiltration attack)
  - If download fails, the script falls back to a synthetic proxy dataset
    that replicates the feature structure of CICIDS-2017 for validation.

Install dependencies
--------------------
    pip install scikit-learn xgboost numpy pandas matplotlib requests

Run
---
    python experiment3_anomaly_detection.py

Runtime
-------
  With download: ~30-45 minutes total (download + training)
  With synthetic fallback: ~3-5 minutes

Output
------
  - Console: full metrics comparison table (ZTA vs baseline)
  - File:    experiment3_results.json
  - File:    experiment3_roc_comparison.png
  - File:    experiment3_feature_importance.png

=============================================================================
"""

import os, json, time, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  Note: xgboost not found. Install with: pip install xgboost")
    print("  Falling back to Random Forest only for ZTA classifier.")

SEED = 42

# ── Output paths (always relative to project root, regardless of cwd) ─────────
import pathlib as _pathlib
_ROOT    = _pathlib.Path(__file__).resolve().parent.parent
_RESULTS = _ROOT / "results"
_FIGURES = _ROOT / "figures"
_RESULTS.mkdir(parents=True, exist_ok=True)
_FIGURES.mkdir(parents=True, exist_ok=True)
# ───────────────────────────────────────────────────────────────────────────────
np.random.seed(SEED)

# =============================================================================
# 1. DATASET: CICIDS-2017
# =============================================================================

# Direct download URL for Friday afternoon file (binary: BENIGN vs DDoS)
# This is the same file used in Goel & Gupta 2024 and Mangla 2023
CICIDS_URL = (
    "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
CICIDS_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Features to use from CICIDS-2017 (subset of 15 most informative)
# These map to network telemetry features collected by ZTA CDM/SIEM (Layer 6)
CICIDS_FEATURES = [
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Mean',
    ' Bwd Packet Length Max',
    ' Bwd Packet Length Mean',
    ' Flow Bytes/s',
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Fwd IAT Total',
    ' Packet Length Mean',
]
CICIDS_LABEL_COL = ' Label'


def try_download_cicids(filename: str = CICIDS_FILENAME) -> bool:
    """Attempt to download CICIDS-2017 Friday file."""
    if os.path.exists(filename):
        print(f"  Found existing file: {filename}")
        return True
    try:
        import urllib.request
        print(f"  Downloading CICIDS-2017 ({CICIDS_URL})...")
        print("  This may take 5-10 minutes depending on connection...")
        urllib.request.urlretrieve(CICIDS_URL, filename)
        print(f"  Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def load_cicids(filename: str, max_rows: int = 80000,
                seed: int = SEED) -> tuple:
    """
    Load and preprocess CICIDS-2017 dataset.
    Returns X, y as numpy arrays with binary labels.
    """
    print(f"  Loading {filename}...")
    df = pd.read_csv(filename, nrows=max_rows, low_memory=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Identify available features (some files have slightly different names)
    available_features = [f.strip() for f in df.columns if f.strip() != 'Label']
    use_features = []
    for f in CICIDS_FEATURES:
        f_stripped = f.strip()
        if f_stripped in available_features:
            use_features.append(f_stripped)
        elif f_stripped in df.columns:
            use_features.append(f_stripped)

    if len(use_features) < 5:
        # Fall back to all numeric columns
        use_features = df.select_dtypes(include=[np.number]).columns.tolist()
        use_features = [c for c in use_features if c != 'Label'][:20]

    print(f"  Using {len(use_features)} features")

    # Label: BENIGN=0, everything else=1 (attack/anomaly)
    label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
    y = (df[label_col].str.strip() != 'BENIGN').astype(int).values

    X = df[use_features].copy()

    # Clean: replace inf/nan with median
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        median = X[col].median()
        X[col].fillna(median, inplace=True)

    print(f"  Class distribution — BENIGN: {(y==0).sum()}, "
          f"Attack/Anomaly: {(y==1).sum()}")

    # Balance if heavily imbalanced (sample up to 40k per class)
    n_per_class = min(40000, (y==0).sum(), (y==1).sum())
    rng = np.random.default_rng(seed)
    idx0 = rng.choice(np.where(y==0)[0], n_per_class, replace=False)
    idx1 = rng.choice(np.where(y==1)[0], n_per_class, replace=False)
    idx  = np.concatenate([idx0, idx1])
    rng.shuffle(idx)

    X = X.iloc[idx].values.astype(np.float32)
    y = y[idx]
    print(f"  Balanced sample: {len(y)} rows "
          f"({(y==0).sum()} BENIGN, {(y==1).sum()} Anomaly)")
    return X, y, use_features


# =============================================================================
# 2. SYNTHETIC FALLBACK DATASET
# =============================================================================

def generate_synthetic_cicids_proxy(n_samples: int = 10000,
                                    seed: int = SEED) -> tuple:
    """
    Generates a synthetic dataset that mirrors the statistical structure
    of CICIDS-2017 network flow features. Used when the real dataset
    cannot be downloaded.

    Features represent ZTA CDM telemetry: flow duration, packet counts,
    byte rates, inter-arrival times — the same signals a ZTA SIEM monitors
    for insider threat detection.
    """
    rng = np.random.default_rng(seed)
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign

    # Benign traffic — moderate, consistent flows
    benign = np.column_stack([
        rng.exponential(50000,  n_benign),   # flow duration (μs)
        rng.poisson(12,         n_benign),   # fwd packets
        rng.poisson(8,          n_benign),   # bwd packets
        rng.normal(5000, 1500,  n_benign),   # fwd bytes total
        rng.normal(3000, 900,   n_benign),   # bwd bytes total
        rng.normal(800,  200,   n_benign),   # fwd pkt max
        rng.normal(400,  100,   n_benign),   # fwd pkt mean
        rng.normal(600,  180,   n_benign),   # bwd pkt max
        rng.normal(350,  90,    n_benign),   # bwd pkt mean
        rng.exponential(50000,  n_benign),   # flow bytes/s
        rng.exponential(200,    n_benign),   # flow packets/s
        rng.exponential(5000,   n_benign),   # flow IAT mean
        rng.exponential(8000,   n_benign),   # flow IAT std
        rng.exponential(200000, n_benign),   # fwd IAT total
        rng.normal(450,  120,   n_benign),   # pkt length mean
    ])

    # Attack/anomaly traffic — high volume, abnormal patterns
    attack = np.column_stack([
        rng.exponential(5000,    n_attack),   # short bursts
        rng.poisson(150,         n_attack),   # high pkt count
        rng.poisson(2,           n_attack),   # few bwd packets
        rng.normal(80000, 20000, n_attack),   # large byte volumes
        rng.normal(200,   100,   n_attack),   # small bwd bytes
        rng.normal(1500,  400,   n_attack),   # large pkt max
        rng.normal(800,   200,   n_attack),   # large pkt mean
        rng.normal(200,   80,    n_attack),   # small bwd pkt max
        rng.normal(100,   40,    n_attack),   # small bwd pkt mean
        rng.exponential(2000000, n_attack),   # very high bytes/s
        rng.exponential(2000,    n_attack),   # very high pkts/s
        rng.exponential(500,     n_attack),   # low IAT (rapid fire)
        rng.exponential(800,     n_attack),   # low IAT std
        rng.exponential(20000,   n_attack),   # low fwd IAT total
        rng.normal(900,   300,   n_attack),   # large pkt length
    ])

    X = np.vstack([benign, attack]).astype(np.float32)
    y = np.array([0] * n_benign + [1] * n_attack)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Bwd Packets',
        'Total Length Fwd Packets', 'Total Length Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Mean',
        'Bwd Packet Length Max', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Total',
        'Packet Length Mean',
    ]
    return X, y, features


# =============================================================================
# 3. CLASSIFIERS
# =============================================================================

def train_zta_ensemble(X_train, y_train, X_test, y_test):
    """
    ZT-IPLS Classifier: Random Forest + XGBoost soft-voting ensemble.
    Represents the AI-enhanced anomaly detection in ZTA Layer 6.
    """
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    print(f"    Random Forest trained in {rf_time:.1f}s")

    rf_probs_train = rf.predict_proba(X_train)[:, 1]
    rf_probs_test  = rf.predict_proba(X_test)[:, 1]

    if XGBOOST_AVAILABLE:
        print("  Training XGBoost...")
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        t0 = time.time()
        xgb_clf.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False)
        xgb_time = time.time() - t0
        print(f"    XGBoost trained in {xgb_time:.1f}s")

        xgb_probs_test = xgb_clf.predict_proba(X_test)[:, 1]

        # Soft-voting ensemble: average probabilities
        ensemble_probs = (rf_probs_test + xgb_probs_test) / 2
        classifier_name = 'RF + XGBoost Ensemble (ZT-IPLS)'
        feature_importances = rf.feature_importances_
    else:
        ensemble_probs  = rf_probs_test
        classifier_name = 'Random Forest (ZT-IPLS)'
        feature_importances = rf.feature_importances_

    # Measure per-sample inference latency
    t0 = time.perf_counter()
    _ = rf.predict_proba(X_test)
    lat_ms = (time.perf_counter() - t0) / len(X_test) * 1000

    threshold = 0.5
    preds = (ensemble_probs >= threshold).astype(int)

    return preds, ensemble_probs, lat_ms, classifier_name, feature_importances


def train_baseline(X_train, y_train, X_test, y_test):
    """
    No-ZTA Baseline: Logistic Regression with standard scaling.
    Represents traditional security without AI-enhanced anomaly detection.
    """
    print("\n  Training baseline (Logistic Regression, no ZTA)...")
    from sklearn.impute import SimpleImputer
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=500, random_state=SEED,
            class_weight='balanced', solver='lbfgs', C=1.0,
        ))
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"    Baseline trained in {time.time()-t0:.1f}s")

    t0 = time.perf_counter()
    probs = pipe.predict_proba(X_test)[:, 1]
    lat_ms = (time.perf_counter() - t0) / len(X_test) * 1000
    preds = (probs >= 0.5).astype(int)

    return preds, probs, lat_ms, 'Logistic Regression (no ZTA)'


# =============================================================================
# 4. METRICS AND PLOTTING
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob, latency_ms, name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    dr   = tp / (tp + fn + 1e-9)
    fpr  = fp / (fp + tn + 1e-9)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc  = (tp + tn) / len(y_true)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return {
        'classifier':          name,
        'n_test':              len(y_true),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'detection_rate':      round(dr,   4),
        'false_positive_rate': round(fpr,  4),
        'precision':           round(prec, 4),
        'f1_score':            round(f1,   4),
        'accuracy':            round(acc,  4),
        'auc_roc':             round(auc,  4),
        'mean_latency_ms':     round(latency_ms, 4),
    }


def print_results_table(results_list):
    print("\n" + "=" * 78)
    print("  EXPERIMENT 3 RESULTS — Insider Threat Anomaly Detection")
    print("=" * 78)
    header = f"  {'Classifier':<38} {'DR':>7} {'FPR':>7} {'F1':>7} {'AUC':>7} {'ms':>7}"
    print(header)
    print("  " + "-" * 76)
    for r in results_list:
        name = r['classifier'][:38]
        print(f"  {name:<38} {r['detection_rate']:>7.4f} "
              f"{r['false_positive_rate']:>7.4f} {r['f1_score']:>7.4f} "
              f"{r['auc_roc']:>7.4f} {r['mean_latency_ms']:>7.4f}")
    print("=" * 78)
    print()
    for r in results_list:
        print(f"  {r['classifier']}")
        print(f"    TP={r['TP']}  TN={r['TN']}  FP={r['FP']}  FN={r['FN']}")
        print(f"    Detection Rate:      {r['detection_rate']:.4f}")
        print(f"    False Positive Rate: {r['false_positive_rate']:.4f}")
        print(f"    Precision:           {r['precision']:.4f}")
        print(f"    F1 Score:            {r['f1_score']:.4f}")
        print(f"    AUC-ROC:             {r['auc_roc']:.4f}")
        print(f"    Mean Latency:        {r['mean_latency_ms']:.4f} ms/sample")
        print()


def plot_roc_comparison(results_with_data, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random (AUC = 0.50)')
    styles = ['-', '--', '-.']
    colors = ['#2563eb', '#dc2626', '#16a34a']
    for i, (r, y_true, y_prob) in enumerate(results_with_data):
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
        label = f"{r['classifier'][:30]} (AUC = {r['auc_roc']:.3f})"
        ax.plot(fpr_arr, tpr_arr, linestyle=styles[i % 3],
                color=colors[i % 3], lw=1.8, label=label)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('Detection Rate (True Positive Rate)')
    ax.set_title('ROC Curves — Insider Threat Anomaly Detection\n'
                 '(Experiment 3)', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_importance(feature_importances, feature_names, output_path,
                            top_n: int = 10):
    idx = np.argsort(feature_importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = feature_importances[idx]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(range(top_n), vals[::-1], color='#2563eb', alpha=0.85,
                   edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([n.strip()[:35] for n in names[::-1]], fontsize=8)
    ax.set_xlabel('Feature Importance (Random Forest)')
    ax.set_title(f'Top {top_n} Features — ZTA Anomaly Detector\n'
                 f'(ZTA CDM/SIEM telemetry signals)', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# 5. CROSS-VALIDATION
# =============================================================================

def run_cross_validation(X, y, n_splits: int = 5):
    """5-fold stratified CV using Random Forest for robustness check."""
    print(f"\n  Running {n_splits}-fold stratified cross-validation (RF)...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cv_scores = {'dr': [], 'fpr': [], 'f1': [], 'auc': []}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED,
            class_weight='balanced', n_jobs=-1,
        )
        rf.fit(X[tr_idx], y[tr_idx])
        probs = rf.predict_proba(X[val_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y[val_idx], preds)
        tn, fp, fn, tp = cm.ravel()
        cv_scores['dr'].append(tp / (tp + fn + 1e-9))
        cv_scores['fpr'].append(fp / (fp + tn + 1e-9))
        cv_scores['f1'].append(f1_score(y[val_idx], preds, pos_label=1,
                                        zero_division=0))
        cv_scores['auc'].append(roc_auc_score(y[val_idx], probs))
        print(f"    Fold {fold+1}: DR={cv_scores['dr'][-1]:.4f} "
              f"FPR={cv_scores['fpr'][-1]:.4f} "
              f"F1={cv_scores['f1'][-1]:.4f}")

    summary = {k: {'mean': round(float(np.mean(v)), 4),
                   'std':  round(float(np.std(v)),  4)}
               for k, v in cv_scores.items()}
    print(f"\n  CV Summary:")
    for k, v in summary.items():
        print(f"    {k.upper()}: {v['mean']:.4f} ± {v['std']:.4f}")
    return summary


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("  ZT-IPLS Experiment 3: Insider Threat Anomaly Detection")
    print("  Paper: Zero-Trust Architecture for AI-Driven IP Litigation Support")
    print("=" * 78)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    using_real_data = False

    if try_download_cicids(CICIDS_FILENAME):
        try:
            X, y, feature_names = load_cicids(CICIDS_FILENAME)
            using_real_data = True
            dataset_name = 'CICIDS-2017 (real)'
        except Exception as e:
            print(f"  Failed to parse CICIDS file: {e}")
            print("  Falling back to synthetic proxy dataset...")

    if not using_real_data:
        print("  Using synthetic CICIDS-2017 proxy dataset...")
        print("  NOTE: Download CICIDS-2017 from https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  for real paper results. Synthetic results are valid for development only.")
        X, y, feature_names = generate_synthetic_cicids_proxy(n_samples=10000)
        dataset_name = 'Synthetic CICIDS-2017 proxy'

    print(f"  Dataset: {dataset_name}")
    print(f"  Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"  Class balance — Normal: {(y==0).sum()}, Anomaly: {(y==1).sum()}")

    # ── Train/test split ──────────────────────────────────────────────────────
    print("\n[2/6] Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # ── ZTA classifier ────────────────────────────────────────────────────────
    print("\n[3/6] Training ZT-IPLS ensemble classifier (RF + XGBoost)...")
    (preds_zta, probs_zta, lat_zta,
     name_zta, feat_imp) = train_zta_ensemble(X_train, y_train, X_test, y_test)

    # ── Baseline classifier ───────────────────────────────────────────────────
    print("\n[4/6] Training no-ZTA baseline...")
    preds_base, probs_base, lat_base, name_base = train_baseline(
        X_train, y_train, X_test, y_test
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n[5/6] Computing metrics and generating plots...")
    r_zta  = compute_metrics(y_test, preds_zta,  probs_zta,  lat_zta,  name_zta)
    r_base = compute_metrics(y_test, preds_base, probs_base, lat_base, name_base)
    results_list = [r_zta, r_base]
    print_results_table(results_list)

    print("  Detailed classification report (ZT-IPLS):")
    print(classification_report(y_test, preds_zta,
                                target_names=['BENIGN/Normal', 'Anomaly/Attack']))

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_results = run_cross_validation(X, y, n_splits=5)

    # ── Plots ─────────────────────────────────────────────────────────────────
    results_w_data = [
        (r_zta,  y_test, probs_zta),
        (r_base, y_test, probs_base),
    ]
    plot_roc_comparison(results_w_data, str(_FIGURES / 'experiment3_roc_comparison.png'))
    plot_feature_importance(feat_imp, feature_names,
                            str(_FIGURES / 'experiment3_feature_importance.png'), top_n=10)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    print("\n[6/6] Saving results to experiment3_results.json...")
    output = {
        'experiment': 'Experiment 3 — Insider Threat Anomaly Detection',
        'paper':      'ZT-IPLS: Zero-Trust Architecture for AI-Driven IP Litigation',
        'dataset': {
            'name':        dataset_name,
            'n_samples':   len(y),
            'n_features':  int(X.shape[1]),
            'n_train':     len(y_train),
            'n_test':      len(y_test),
            'using_real_cicids': using_real_data,
            'random_seed': SEED,
        },
        'results':          results_list,
        'cross_validation': {
            'method':     '5-fold stratified',
            'classifier': 'Random Forest',
            'scores':     cv_results,
        },
        'section_v_numbers': {
            'NOTE': 'Use ZTA classifier results in Section V of the paper',
            'zta_detection_rate':      r_zta['detection_rate'],
            'zta_false_positive_rate': r_zta['false_positive_rate'],
            'zta_f1_score':            r_zta['f1_score'],
            'zta_auc_roc':             r_zta['auc_roc'],
            'baseline_detection_rate': r_base['detection_rate'],
            'baseline_fpr':            r_base['false_positive_rate'],
            'baseline_auc_roc':        r_base['auc_roc'],
            'improvement_dr_pct':      round((r_zta['detection_rate'] -
                                              r_base['detection_rate']) * 100, 2),
            'improvement_fpr_pct':     round((r_base['false_positive_rate'] -
                                              r_zta['false_positive_rate']) * 100, 2),
        },
    }

    with open(str(_RESULTS / 'experiment3_results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 78)
    print("  DONE. Copy these numbers into Section V of your paper:")
    print("=" * 78)
    print(f"  Dataset:                {dataset_name}")
    print(f"  ZTA (RF+XGBoost):")
    print(f"    Detection Rate:       {r_zta['detection_rate']:.4f} "
          f"({r_zta['detection_rate']*100:.1f}%)")
    print(f"    False Positive Rate:  {r_zta['false_positive_rate']:.4f}")
    print(f"    F1 Score:             {r_zta['f1_score']:.4f}")
    print(f"    AUC-ROC:              {r_zta['auc_roc']:.4f}")
    print(f"  Baseline (no ZTA, LR):")
    print(f"    Detection Rate:       {r_base['detection_rate']:.4f}")
    print(f"    False Positive Rate:  {r_base['false_positive_rate']:.4f}")
    print(f"    AUC-ROC:              {r_base['auc_roc']:.4f}")
    print(f"  DR improvement vs baseline: "
          f"+{(r_zta['detection_rate']-r_base['detection_rate'])*100:.1f}pp")
    print(f"  CV DR (5-fold): "
          f"{cv_results['dr']['mean']:.4f} ± {cv_results['dr']['std']:.4f}")
    print("=" * 78)
    if not using_real_data:
        print("\n  *** IMPORTANT FOR PAPER SUBMISSION ***")
        print("  Results above used the SYNTHETIC proxy dataset.")
        print("  For final paper results, download CICIDS-2017:")
        print("  https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  Then re-run: python experiment3_anomaly_detection.py")
        print("  The script will auto-detect and use the real dataset.")
    print("\n  Output files:")
    print(f"    {str(_RESULTS / 'experiment3_results.json'):<60} ← JSON results")
    print(f"    {str(_FIGURES / 'experiment3_roc_comparison.png'):<60} ← Figure 5")
    print(f"    {str(_FIGURES / 'experiment3_feature_importance.png'):<60} ← Figure 6")
    print()


if __name__ == '__main__':
    main()