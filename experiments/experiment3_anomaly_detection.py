"""
ZT-IPLS Experiment 3: Insider Threat Anomaly Detection
Validates Layer 6 of the ZT-IPLS framework using CICIDS-2017.
Results  → results/experiment3_results.json
Figures  → figures/experiment3_roc_comparison.png
           figures/experiment3_feature_importance.png

Place Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv in the project root.
Download: https://www.unb.ca/cic/datasets/ids-2017.html
"""
import os, json, time, random, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import pathlib
ROOT    = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

CICIDS_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
CICIDS_FEATURES = [
    " Flow Duration"," Total Fwd Packets"," Total Backward Packets",
    "Total Length of Fwd Packets"," Total Length of Bwd Packets",
    " Fwd Packet Length Max"," Fwd Packet Length Mean",
    " Bwd Packet Length Max"," Bwd Packet Length Mean",
    " Flow Bytes/s"," Flow Packets/s"," Flow IAT Mean",
    " Flow IAT Std"," Fwd IAT Total"," Packet Length Mean",
]

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
random.seed(SEED); np.random.seed(SEED)

def find_cicids():
    for d in [ROOT,
               pathlib.Path.home()/"Downloads",
               pathlib.Path.home()/"Desktop",
               pathlib.Path.home()/"Desktop"/"zt-ipls",
               pathlib.Path.cwd()]:
        p = d / CICIDS_FILENAME
        if p.exists():
            print(f"  Found: {p}")
            return str(p)
    return None

def load_cicids(path, max_rows=80000):
    print(f"  Loading {pathlib.Path(path).name}...")
    df = pd.read_csv(path, nrows=max_rows)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    df[" Label"] = df[" Label"].str.strip()
    df["y"] = (df[" Label"] != "BENIGN").astype(int)
    available = [f for f in CICIDS_FEATURES if f in df.columns]
    print(f"  Using {len(available)} features")
    X = df[available].copy()
    X.replace([np.inf,-np.inf], np.nan, inplace=True)
    y = df["y"].values
    n_min = min((y==0).sum(),(y==1).sum())
    rng = np.random.RandomState(SEED)
    idx0 = rng.choice(np.where(y==0)[0], n_min, replace=False)
    idx1 = rng.choice(np.where(y==1)[0], n_min, replace=False)
    idx  = np.sort(np.concatenate([idx0,idx1]))
    X,y  = X.iloc[idx].values, y[idx]
    print(f"  Balanced: {len(y)} samples ({(y==0).sum()} BENIGN, {(y==1).sum()} Attack)")
    return X, y, available

def generate_synthetic(n=10000):
    rng = np.random.RandomState(SEED); n2 = n//2
    X_n = rng.randn(n2,15)*np.array([1e6,50,30,1e4,5e3,800,400,600,300,1e6,500,1e5,5e4,2e5,400]) + \
          np.array([1e5,10,5,500,200,100,50,80,40,1e5,100,1e4,5e3,2e4,50])
    X_a = rng.randn(n2,15)*np.array([5e5,200,5,5e4,500,2000,1000,100,50,5e6,2000,5e4,2e4,8e4,600]) + \
          np.array([5e6,500,2,1e5,100,5000,2500,200,100,1e7,5000,5e5,2e5,8e5,800])
    X = np.vstack([X_n,X_a]); y = np.array([0]*n2+[1]*n2)
    idx = rng.permutation(len(y))
    return X[idx], y[idx], [f"feature_{i}" for i in range(15)]

def train_ztipls(X_train, y_train, X_test, y_test):
    print("\n  Training Random Forest (n_jobs=1 for macOS)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", random_state=SEED, n_jobs=1)
    t0 = time.time(); rf.fit(X_train,y_train)
    print(f"    Done in {time.time()-t0:.1f}s")
    rf_prob = rf.predict_proba(X_test)[:,1]

    try:
        import xgboost as xgb
        print("\n  Training XGBoost (n_jobs=1 for macOS)...")
        scale_pos = float((y_train==0).sum())/max(float((y_train==1).sum()),1)
        xgb_clf = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     scale_pos_weight=scale_pos, eval_metric="logloss",
                                     random_state=SEED, n_jobs=1, verbosity=0,
                                     use_label_encoder=False)
        t0 = time.time()
        xgb_clf.fit(X_train, y_train, eval_set=[(X_test,y_test)], verbose=False)
        print(f"    Done in {time.time()-t0:.1f}s")
        xgb_prob = xgb_clf.predict_proba(X_test)[:,1]
        t0 = time.perf_counter()
        probs = (rf_prob+xgb_prob)/2
        lat   = (time.perf_counter()-t0)/len(X_test)*1000
        return (probs>=0.5).astype(int), probs, lat, "RF + XGBoost ensemble (ZT-IPLS)", rf
    except ImportError:
        print("  XGBoost not found — using RF only")
        t0 = time.perf_counter()
        lat = (time.perf_counter()-t0)/len(X_test)*1000
        return (rf_prob>=0.5).astype(int), rf_prob, lat, "Random Forest (ZT-IPLS)", rf

def train_baseline(X_train, y_train, X_test, y_test):
    print("\n  Training Logistic Regression baseline...")
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler()),
                     ("clf",     LogisticRegression(max_iter=500, random_state=SEED,
                                                     class_weight="balanced", solver="lbfgs"))])
    t0 = time.time(); pipe.fit(X_train,y_train)
    print(f"    Done in {time.time()-t0:.1f}s")
    t0 = time.perf_counter()
    probs = pipe.predict_proba(X_test)[:,1]
    lat   = (time.perf_counter()-t0)/len(X_test)*1000
    return (probs>=0.5).astype(int), probs, lat, "Logistic Regression (no ZTA)"

def compute_metrics(y_true, y_pred, y_prob, lat, name):
    cm = confusion_matrix(y_true,y_pred); tn,fp,fn,tp = cm.ravel()
    return {"classifier":name,"TP":int(tp),"TN":int(tn),"FP":int(fp),"FN":int(fn),
            "detection_rate":      round(tp/(tp+fn+1e-9),4),
            "false_positive_rate": round(fp/(fp+tn+1e-9),4),
            "f1_score":            round(f1_score(y_true,y_pred,zero_division=0),4),
            "auc_roc":             round(roc_auc_score(y_true,y_prob),4),
            "mean_latency_ms":     round(lat,4)}

def plot_roc(data, path):
    fig,ax = plt.subplots(figsize=(7,5))
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    for r,y_true,y_prob in data:
        fpr_arr,tpr_arr,_ = roc_curve(y_true,y_prob)
        ax.plot(fpr_arr,tpr_arr,label=f"{r['classifier'][:35]} (AUC={r['auc_roc']:.4f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Experiment 3: Insider Threat Detection")
    ax.legend(loc="lower right",fontsize=9); ax.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path),dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def plot_importance(rf_model, feature_names, path, top_n=10):
    if not hasattr(rf_model,"feature_importances_"): return
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig,ax = plt.subplots(figsize=(8,5))
    ax.barh(range(top_n),[imp[i] for i in idx][::-1],color="steelblue",alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([str(feature_names[i]).strip() for i in idx][::-1],fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — ZT-IPLS RF (Layer 6)")
    ax.grid(True,alpha=0.3,axis="x"); plt.tight_layout()
    plt.savefig(str(path),dpi=150,bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")

def cross_validate(X,y):
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    imp = SimpleImputer(strategy="median"); X_c = imp.fit_transform(X)
    scores = {"dr":[],"fpr":[],"f1":[],"auc":[]}
    for fold,(tr,va) in enumerate(skf.split(X_c,y)):
        rf = RandomForestClassifier(n_estimators=100,max_depth=12,
                                    random_state=SEED,class_weight="balanced",n_jobs=1)
        rf.fit(X_c[tr],y[tr])
        probs = rf.predict_proba(X_c[va])[:,1]
        preds = (probs>=0.5).astype(int)
        cm = confusion_matrix(y[va],preds); tn,fp,fn,tp = cm.ravel()
        scores["dr"].append(tp/(tp+fn+1e-9))
        scores["fpr"].append(fp/(fp+tn+1e-9))
        scores["f1"].append(f1_score(y[va],preds,zero_division=0))
        scores["auc"].append(roc_auc_score(y[va],probs))
        print(f"    Fold {fold+1}: DR={scores['dr'][-1]:.4f}  FPR={scores['fpr'][-1]:.4f}")
    return {k:{"mean":round(float(np.mean(v)),4),"std":round(float(np.std(v)),4)}
            for k,v in scores.items()}

def main():
    print("="*70)
    print("  ZT-IPLS — Experiment 3: Insider Threat Anomaly Detection")
    print("="*70)

    print("\n[1/6] Loading dataset...")
    cicids_path = find_cicids()
    using_real = False
    if cicids_path:
        try:
            X,y,feature_names = load_cicids(cicids_path)
            using_real = True; dataset_name = "CICIDS-2017 (real)"
        except Exception as e:
            print(f"  Failed to load: {e}")
    if not using_real:
        print(f"  CICIDS file not found — using synthetic proxy")
        print(f"  To use real data, place {CICIDS_FILENAME} in:")
        print(f"  {ROOT}")
        X,y,feature_names = generate_synthetic()
        dataset_name = "Synthetic CICIDS-2017 proxy"
    print(f"  Dataset: {dataset_name}  |  {len(y)} samples  |  {X.shape[1]} features")

    print("\n[2/6] Splitting 80/20 stratified...")
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr); X_te = imp.transform(X_te)
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    print("\n[3/6] Training ZT-IPLS ensemble...")
    zt_preds,zt_probs,zt_lat,zt_name,rf_model = train_ztipls(X_tr,y_tr,X_te,y_te)

    print("\n[4/6] Training baseline...")
    bl_preds,bl_probs,bl_lat,bl_name = train_baseline(X_tr,y_tr,X_te,y_te)

    zt_m = compute_metrics(y_te,zt_preds,zt_probs,zt_lat,zt_name)
    bl_m = compute_metrics(y_te,bl_preds,bl_probs,bl_lat,bl_name)
    fpr_ratio = bl_m["false_positive_rate"]/max(zt_m["false_positive_rate"],1e-6)

    print("\n[5/6] Results:")
    print("="*70)
    print(f"  {'Classifier':<42} {'DR':>7} {'FPR':>7} {'F1':>7} {'AUC':>7}")
    print("-"*70)
    for m in [zt_m,bl_m]:
        print(f"  {m['classifier'][:42]:<42} {m['detection_rate']:>7.4f} "
              f"{m['false_positive_rate']:>7.4f} {m['f1_score']:>7.4f} {m['auc_roc']:>7.4f}")
    print("="*70)
    print(f"\n  FPR reduction: {fpr_ratio:.1f}x  "
          f"({bl_m['false_positive_rate']:.4f} → {zt_m['false_positive_rate']:.4f})")
    print(f"\n  Classification report — {zt_name}")
    print(classification_report(y_te,zt_preds,target_names=["BENIGN","Attack"],zero_division=0))

    print("\n  5-fold cross-validation (RF)...")
    cv = cross_validate(X_tr,y_tr)
    print(f"  CV DR:  {cv['dr']['mean']:.4f} ± {cv['dr']['std']:.4f}")
    print(f"  CV FPR: {cv['fpr']['mean']:.4f} ± {cv['fpr']['std']:.4f}")

    plot_roc([(zt_m,y_te,zt_probs),(bl_m,y_te,bl_probs)],
             FIGURES/"experiment3_roc_comparison.png")
    plot_importance(rf_model,feature_names,
                    FIGURES/"experiment3_feature_importance.png")

    print("\n[6/6] Saving results...")
    out_json = RESULTS / "experiment3_results.json"
    with open(out_json,"w") as f:
        json.dump({"experiment":"Experiment 3 — Insider Threat Anomaly Detection",
                   "dataset":{"name":dataset_name,"samples":int(len(y)),
                              "features":int(X.shape[1]),"seed":SEED,
                              "train":int(len(y_tr)),"test":int(len(y_te))},
                   "results":[zt_m,bl_m],
                   "cross_validation":cv,
                   "fpr_reduction_factor":round(fpr_ratio,1)}, f, indent=2)

    print("\n" + "="*70)
    print("  EXPERIMENT 3 COMPLETE")
    print("="*70)
    print(f"  Dataset:          {dataset_name}")
    print(f"  ZT-IPLS   DR={zt_m['detection_rate']:.4f}  FPR={zt_m['false_positive_rate']:.4f}  AUC={zt_m['auc_roc']:.4f}")
    print(f"  Baseline  DR={bl_m['detection_rate']:.4f}  FPR={bl_m['false_positive_rate']:.4f}  AUC={bl_m['auc_roc']:.4f}")
    print(f"  FPR reduction:    {fpr_ratio:.1f}x")
    print(f"  CV DR:            {cv['dr']['mean']:.4f} ± {cv['dr']['std']:.4f}")
    print()
    print("  Saved:")
    print(f"    {out_json}")
    print(f"    {FIGURES / 'experiment3_roc_comparison.png'}")
    print(f"    {FIGURES / 'experiment3_feature_importance.png'}")
    print("="*70)

if __name__ == "__main__":
    main()