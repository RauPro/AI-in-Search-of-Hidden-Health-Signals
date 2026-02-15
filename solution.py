
"""
=============================================================================
 HEA HACKATHON: "AI in Search of Hidden Health Signals"
 Gold-Level Disease Onset Prediction — RAND HRS Longitudinal Data
=============================================================================
 Predicts onset of chronic disease from self-reported health, lifestyle,
 and socioeconomic signals in currently-healthy individuals.

 Scoring: F2-Score, PR-AUC, ROC-AUC
 Compliance: No data leakage, fairness, explainability, open-source only
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, fbeta_score, precision_recall_curve, auc,
    classification_report, confusion_matrix
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed. pip install shap")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH = "randhrs1992_2022v1.sas7bdat"
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

# Waves: start at 3 so we can compute lag from wave 2
WAVES = list(range(3, 17))

# ── Variable dictionaries (verified to exist in dataset) ─────────────────
# Maps friendly_name → RAND column suffix  (columns = R{wave}{suffix})

# FEATURES — safe to use as predictors
FEATURE_VARS = {
    # Self-rated health
    "Health":         "SHLT",       # 1=Excellent .. 5=Poor
    "HealthChange":   "SHLTC",      # Health change perception

    # Body
    "BMI":            "BMI",

    # Mental health
    "Depression":     "CESD",       # CES-D 8-item (0-8)
    "Psych":          "PSYCH",      # Psychiatric conditions flag

    # Functional limitations (all verified to exist wave 2-16)
    "Mobility":       "MOBILA",     # Mobility limitations count
    "LgMuscle":       "LGMUSA",     # Large muscle group limitations
    "GrossMotor":     "GROSSA",     # Gross motor limitations
    "FineMotor":      "FINEA",      # Fine motor limitations

    # Cognition (waves 3-13)
    "WordRecall":     "TR20",       # Total word recall
    "MentalStatus":   "MSTOT",      # Mental status summary

    # Health behaviors
    "EverSmoked":     "SMOKEV",     # Ever smoked (0/1)
    "SmokeNow":       "SMOKEN",     # Smoke currently (0/1)
    "Drinks":         "DRINK",      # Drinks per day
    "VigExercise":    "VGACTX",     # Vigorous exercise (0/1, waves 7-16)

    # Healthcare utilization
    "Hospital":       "HOSP",       # Hospital stay since last wave (0/1)
    "DoctorVisits":   "DOCTIM",     # Number of doctor visits
    "OOPMedical":     "OOPMD",      # Out-of-pocket medical expenses

    # Blood pressure as RISK FACTOR (not target)
    "HighBP":         "HIBPE",      # Ever had high blood pressure

    # Other risk
    "ConditionCount": "CONDE",      # Count of chronic conditions (composite)

    # Age
    "Age":            "AGEY_E",

    # Socioeconomic
    "Working":        "WORK",       # Currently working (0/1)
    "MaritalStatus":  "MSTAT",      # Marital status code
}

# TARGETS — used only to build Y labels, NEVER as features
TARGET_VARS = {
    "T_Diabetes":  "DIABE",     # Ever had diabetes
    "T_Heart":     "HEARTE",    # Ever had heart problems
    "T_Stroke":    "STROKE",    # Ever had stroke
    "T_Lung":      "LUNGE",     # Ever had lung disease
    "T_Cancer":    "CANCRE",    # Ever had cancer
    "T_Arthritis": "ARTHRE",    # Ever had arthritis
}

# Static demographics
STATIC_VARS = ["HHIDPN", "RAGENDER", "RARACEM", "RAHISPAN", "RAEDYRS"]

def banner(msg):
    print(f"\n{'='*70}\n  {msg}\n{'='*70}\n")



# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: LOAD & RESHAPE
# ═══════════════════════════════════════════════════════════════════════════

def load_and_reshape(data_path):
    banner("PHASE 1: DATA LOADING & VARIABLE EXTRACTION")
    print(f"Loading: {data_path}")
    df_raw = pd.read_sas(str(data_path))
    print(f"Raw data: {df_raw.shape[0]:,} x {df_raw.shape[1]:,}")

    # Static demographics
    avail_static = [c for c in STATIC_VARS if c in df_raw.columns]
    df_static = df_raw[avail_static].copy()
    print(f"Static vars: {avail_static}")

    all_vars = {**FEATURE_VARS, **TARGET_VARS}
    long_frames = []

    for wave in WAVES:
        wave_data = {"HHIDPN": df_raw["HHIDPN"]}
        found = 0
        for name, suffix in all_vars.items():
            col = f"R{wave}{suffix}"
            if col in df_raw.columns:
                wave_data[name] = df_raw[col].values
                found += 1
            else:
                wave_data[name] = np.nan
        if found < 5:
            print(f"  Wave {wave}: {found} vars — SKIPPED")
            continue
        df_w = pd.DataFrame(wave_data)
        df_w["Wave"] = wave
        long_frames.append(df_w)
        print(f"  Wave {wave}: {found}/{len(all_vars)} vars")

    df_long = pd.concat(long_frames, ignore_index=True)
    df_long = pd.merge(df_long, df_static, on="HHIDPN", how="left")
    df_long.sort_values(["HHIDPN", "Wave"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    print(f"\nLong format: {df_long.shape[0]:,} rows x {df_long.shape[1]} cols")
    return df_long


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    banner("PHASE 2: FEATURE ENGINEERING")
    df = df.sort_values(["HHIDPN", "Wave"]).copy()

    feat_cols = list(FEATURE_VARS.keys())
    tgt_cols = list(TARGET_VARS.keys())

    # Forward-fill within person
    print("Forward-filling missing values...")
    for c in feat_cols + tgt_cols:
        if c in df.columns:
            df[c] = df.groupby("HHIDPN")[c].ffill()

    # ── Build TARGET ──────────────────────────────────────────────────────
    print("Building target: disease onset...")
    for tc in tgt_cols:
        if tc in df.columns:
            df[tc] = df[tc].fillna(0).clip(0, 1)

    df["CurrentDiseaseCount"] = df[tgt_cols].sum(axis=1)
    df["IsSick"] = (df["CurrentDiseaseCount"] > 0).astype(int)
    df["NextSick"] = df.groupby("HHIDPN")["IsSick"].shift(-1)

    # Onset: healthy now → sick next wave
    df["Y"] = 0
    df.loc[(df["IsSick"] == 0) & (df["NextSick"] == 1), "Y"] = 1

    # Filter: only at-risk (currently healthy + known future)
    df = df[(df["IsSick"] == 0) & (df["NextSick"].notna())].copy()
    print(f"At-risk cohort: {len(df):,} | Event rate: {df['Y'].mean():.2%}")

    # ── Remove all-NaN feature columns before engineering ─────────────────
    for c in feat_cols[:]:
        if c in df.columns and df[c].isna().all():
            print(f"  Dropping all-NaN column: {c}")
            df.drop(columns=[c], inplace=True)
            feat_cols.remove(c)

    # ── Continuous variables for temporal features ────────────────────────
    continuous = [c for c in [
        "BMI", "Depression", "Mobility", "Health", "Age",
        "LgMuscle", "GrossMotor", "FineMotor", "WordRecall",
        "MentalStatus", "Drinks", "DoctorVisits", "OOPMedical",
        "ConditionCount", "HealthChange"
    ] if c in df.columns]

    print(f"Temporal features on {len(continuous)} continuous vars...")
    for col in continuous:
        g = df.groupby("HHIDPN")[col]
        df[f"D_{col}"] = g.diff()
        df[f"L1_{col}"] = g.shift(1)
        df[f"L2_{col}"] = g.shift(2)
        df[f"A_{col}"] = df.groupby("HHIDPN")[f"D_{col}"].diff()

    # ── Rolling statistics (3-wave) ───────────────────────────────────────
    rolling = [c for c in ["BMI","Depression","Health","Mobility"] if c in df.columns]
    print(f"Rolling stats on {len(rolling)} vars...")
    for col in rolling:
        g = df.groupby("HHIDPN")[col]
        df[f"R3m_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"R3s_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        # Min/max over rolling window
        df[f"R3min_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=1).min())
        df[f"R3max_{col}"] = g.transform(lambda x: x.rolling(3, min_periods=1).max())
        # Range
        df[f"R3range_{col}"] = df[f"R3max_{col}"] - df[f"R3min_{col}"]

    # ── Cross-domain interactions ─────────────────────────────────────────
    print("Cross-domain interactions...")
    def safe_mul(a, b):
        aa = df.get(a)
        bb = df.get(b)
        if aa is not None and bb is not None:
            return aa * bb
        return None

    interactions = {
        "Ix_AgeBMI": ("Age", "BMI"),
        "Ix_AgeDep": ("Age", "Depression"),
        "Ix_AgeHealth": ("Age", "Health"),
        "Ix_DepHealth": ("Depression", "Health"),
        "Ix_BMIMob": ("BMI", "Mobility"),
        "Ix_BMIDep": ("BMI", "Depression"),
        "Ix_AgeMob": ("Age", "Mobility"),
        "Ix_AgeCondCnt": ("Age", "ConditionCount"),
        "Ix_HealthMob": ("Health", "Mobility"),
        "Ix_DepMob": ("Depression", "Mobility"),
    }
    for name, (a, b) in interactions.items():
        r = safe_mul(a, b)
        if r is not None:
            df[name] = r

    # Age non-linearity
    if "Age" in df.columns:
        df["AgeSq"] = df["Age"] ** 2
        # Age bins
        df["AgeBin"] = pd.cut(df["Age"], bins=[0,55,60,65,70,75,80,85,120],
                              labels=list(range(8))).astype(float)

    # BMI categories
    if "BMI" in df.columns:
        df["BMI_Cat"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,35,100],
                               labels=[0,1,2,3,4]).astype(float)

    # ── Threshold flags ───────────────────────────────────────────────────
    print("Threshold flags...")
    if "BMI" in df.columns:
        df["Fl_Obese"] = (df["BMI"] >= 30).astype(int)
        df["Fl_Underweight"] = (df["BMI"] < 18.5).astype(int)
        df["Fl_Overweight"] = (df["BMI"] >= 25).astype(int)
    if "Depression" in df.columns:
        df["Fl_Depressed"] = (df["Depression"] >= 4).astype(int)
        df["Fl_MildDep"] = (df["Depression"] >= 2).astype(int)
    if "Health" in df.columns:
        df["Fl_PoorHealth"] = (df["Health"] >= 4).astype(int)
        df["Fl_FairHealth"] = (df["Health"] >= 3).astype(int)
    if "Mobility" in df.columns:
        l1_mob = df.get("L1_Mobility")
        if l1_mob is not None:
            df["Fl_MobCrisis"] = ((df["Mobility"] > 0) & (l1_mob.fillna(0) == 0)).astype(int)
    if "D_Health" in df.columns:
        df["Fl_HealthDrop"] = (df["D_Health"] >= 2).astype(int)
        df["Fl_HealthDrop1"] = (df["D_Health"] >= 1).astype(int)
    if "D_Depression" in df.columns:
        df["Fl_DepSpike"] = (df["D_Depression"] >= 3).astype(int)
    if "D_BMI" in df.columns:
        df["Fl_BMISurge"] = (df["D_BMI"].abs() >= 3).astype(int)

    # ── Cumulative exposure ───────────────────────────────────────────────
    print("Cumulative exposure...")
    cum_cols = {
        "CumObese": "Fl_Obese",
        "CumDepressed": "Fl_Depressed",
        "CumPoorH": "Fl_PoorHealth",
        "CumHosp": "Hospital",
    }
    for new_name, src in cum_cols.items():
        if src in df.columns:
            df[new_name] = df.groupby("HHIDPN")[src].cumsum()

    # ── Demographics encoding ─────────────────────────────────────────────
    print("Demographics...")
    if "RAGENDER" in df.columns:
        df["IsFemale"] = (df["RAGENDER"] == 2).astype(int)
    if "RARACEM" in df.columns:
        df["RaceWhite"] = (df["RARACEM"] == 1).astype(int)
        df["RaceBlack"] = (df["RARACEM"] == 2).astype(int)
        df["RaceOther"] = (df["RARACEM"] == 3).astype(int)
    if "RAHISPAN" in df.columns:
        df["IsHispanic"] = (df["RAHISPAN"] == 1).astype(int)
    if "RAEDYRS" in df.columns:
        df["Education"] = df["RAEDYRS"]
        df["Fl_LowEduc"] = (df["RAEDYRS"] < 12).astype(int)

    # ── Health trajectory ─────────────────────────────────────────────────
    if "D_Health" in df.columns:
        df["HealthTraj"] = 0
        df.loc[df["D_Health"] > 0.5, "HealthTraj"] = 1
        df.loc[df["D_Health"] < -0.5, "HealthTraj"] = -1

    # ── Marital status encoding ───────────────────────────────────────────
    if "MaritalStatus" in df.columns:
        df["IsMarried"] = (df["MaritalStatus"].isin([1, 2, 3])).astype(int)

    # ── Delta interactions (rate-of-change interactions) ───────────────────
    print("Delta interactions...")
    delta_ixs = {
        "Dix_BMI_Health": ("D_BMI", "D_Health"),
        "Dix_Dep_Health": ("D_Depression", "D_Health"),
        "Dix_BMI_Dep": ("D_BMI", "D_Depression"),
        "Dix_Mob_Health": ("D_Mobility", "D_Health"),
    }
    for name, (a, b) in delta_ixs.items():
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]

    # ── Ratio features ────────────────────────────────────────────────────
    print("Ratio features...")
    if "BMI" in df.columns and "L1_BMI" in df.columns:
        df["BMI_Ratio"] = df["BMI"] / df["L1_BMI"].replace(0, np.nan)
    if "Depression" in df.columns and "L1_Depression" in df.columns:
        df["Dep_Ratio"] = df["Depression"] / df["L1_Depression"].replace(0, np.nan)

    n_feats = len([c for c in df.columns if c not in
                   {"HHIDPN","Wave","Y","NextSick","IsSick",
                    "CurrentDiseaseCount","RAGENDER","RARACEM",
                    "RAHISPAN","RAEDYRS"} | set(tgt_cols)])
    print(f"\n>> Feature engineering complete: {n_feats} features")
    return df



# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def prepare_splits(df):
    banner("PHASE 3: TRAIN / VALIDATION SPLIT")
    tgt_cols = list(TARGET_VARS.keys())
    exclude = {"HHIDPN","Wave","Y","NextSick","IsSick",
               "CurrentDiseaseCount","RAGENDER","RARACEM",
               "RAHISPAN","RAEDYRS","MaritalStatus"} | set(tgt_cols)
    features = sorted([c for c in df.columns if c not in exclude])
    print(f"Feature count: {len(features)}")

    df["Y"] = df["Y"].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(df, df["Y"], groups=df["HHIDPN"]))

    X_tr = df.iloc[train_idx][features].copy()
    y_tr = df.iloc[train_idx]["Y"].copy()
    X_va = df.iloc[val_idx][features].copy()
    y_va = df.iloc[val_idx]["Y"].copy()
    demo = df.iloc[val_idx][["HHIDPN","RAGENDER","RARACEM","Age"]].copy()

    print(f"Train: {len(X_tr):,} | Val: {len(X_va):,}")
    print(f"Train event: {y_tr.mean():.2%} | Val event: {y_va.mean():.2%}")
    return features, X_tr, y_tr, X_va, y_va, demo


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: MULTI-MODEL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def train_models(X_tr, y_tr, X_va, y_va, features):
    banner("PHASE 4: MULTI-MODEL ENSEMBLE")

    neg = (y_tr==0).sum()
    pos = (y_tr==1).sum()
    ratio = neg / max(pos, 1)
    print(f"Imbalance: {neg:,} neg / {pos:,} pos = {ratio:.1f}:1")

    models = {}
    preds = {}

    # ── LightGBM ──────────────────────────────────────────────────────────
    print("\n--- LightGBM ---")
    lgb = LGBMClassifier(
        n_estimators=5000, learning_rate=0.005, max_depth=7, num_leaves=63,
        min_child_samples=30, colsample_bytree=0.6, subsample=0.7,
        subsample_freq=5, reg_alpha=0.1, reg_lambda=3.0,
        scale_pos_weight=ratio,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[
                __import__("lightgbm").early_stopping(300, verbose=True),
                __import__("lightgbm").log_evaluation(500),
            ])
    p_lgb = lgb.predict_proba(X_va)[:, 1]
    models["LightGBM"] = lgb
    preds["LightGBM"] = p_lgb
    print(f"LightGBM ROC-AUC: {roc_auc_score(y_va, p_lgb):.4f}")

    # ── LightGBM v2 (different hyperparameters for diversity) ─────────────
    print("\n--- LightGBM_v2 ---")
    lgb2 = LGBMClassifier(
        n_estimators=5000, learning_rate=0.01, max_depth=5, num_leaves=31,
        min_child_samples=50, colsample_bytree=0.5, subsample=0.8,
        subsample_freq=3, reg_alpha=0.5, reg_lambda=5.0,
        scale_pos_weight=ratio * 0.8,  # slightly less aggressive
        random_state=SEED + 1, n_jobs=-1, verbose=-1,
    )
    lgb2.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
             callbacks=[
                 __import__("lightgbm").early_stopping(300, verbose=True),
                 __import__("lightgbm").log_evaluation(500),
             ])
    p_lgb2 = lgb2.predict_proba(X_va)[:, 1]
    models["LightGBM_v2"] = lgb2
    preds["LightGBM_v2"] = p_lgb2
    print(f"LightGBM_v2 ROC-AUC: {roc_auc_score(y_va, p_lgb2):.4f}")

    # ── CatBoost ──────────────────────────────────────────────────────────
    print("\n--- CatBoost ---")
    cb = CatBoostClassifier(
        iterations=5000, learning_rate=0.01, depth=6, l2_leaf_reg=5,
        loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=ratio,
        early_stopping_rounds=300, verbose=500,
        allow_writing_files=False, random_seed=SEED,
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    p_cb = cb.predict_proba(X_va)[:, 1]
    models["CatBoost"] = cb
    preds["CatBoost"] = p_cb
    print(f"CatBoost ROC-AUC: {roc_auc_score(y_va, p_cb):.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=5000, learning_rate=0.01, max_depth=6,
        min_child_weight=10, colsample_bytree=0.6, subsample=0.7,
        reg_alpha=0.1, reg_lambda=3.0, scale_pos_weight=ratio,
        eval_metric="auc", early_stopping_rounds=300,
        verbosity=0, random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=500)
    p_xgb = xgb.predict_proba(X_va)[:, 1]
    models["XGBoost"] = xgb
    preds["XGBoost"] = p_xgb
    print(f"XGBoost ROC-AUC: {roc_auc_score(y_va, p_xgb):.4f}")

    # ── Stacking Ensemble ─────────────────────────────────────────────────
    print("\n--- Ensemble ---")
    base_names = ["LightGBM", "LightGBM_v2", "CatBoost", "XGBoost"]
    stk_tr = np.column_stack([m.predict_proba(X_tr)[:,1] for m in [models[n] for n in base_names]])
    stk_va = np.column_stack([preds[n] for n in base_names])

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stk_tr, y_tr)
    p_ens = meta.predict_proba(stk_va)[:, 1]
    models["Ensemble_Meta"] = meta
    preds["Ensemble"] = p_ens
    print(f"Ensemble ROC-AUC: {roc_auc_score(y_va, p_ens):.4f}")

    # Simple average
    p_avg = np.mean(stk_va, axis=1)
    preds["Average"] = p_avg
    print(f"Average  ROC-AUC: {roc_auc_score(y_va, p_avg):.4f}")

    # Rank average (robust ensemble)
    from scipy.stats import rankdata
    ranks = np.column_stack([rankdata(preds[n]) for n in base_names])
    p_rank = ranks.mean(axis=1) / len(y_va)
    preds["RankAvg"] = p_rank
    print(f"RankAvg  ROC-AUC: {roc_auc_score(y_va, p_rank):.4f}")

    return models, preds


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: THRESHOLD OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(y_va, predictions):
    banner("PHASE 5: EVALUATION & THRESHOLD OPTIMIZATION")

    results = {}
    best_name = None
    best_f2 = -1

    for name, probs in predictions.items():
        roc = roc_auc_score(y_va, probs)
        prec, rec, _ = precision_recall_curve(y_va, probs)
        pr = auc(rec, prec)

        thresholds = np.arange(0.03, 0.90, 0.005)
        f2s = []
        for t in thresholds:
            p = (probs > t).astype(int)
            f2s.append(fbeta_score(y_va, p, beta=2) if p.sum() > 0 else 0)
        b_f2 = max(f2s)
        b_thr = float(thresholds[np.argmax(f2s)])

        print(f"  {name:<15} ROC={roc:.4f}  PR={pr:.4f}  F2={b_f2:.4f} (t={b_thr:.3f})")

        best_p = (probs > b_thr).astype(int)
        print(classification_report(y_va, best_p,
              target_names=["Healthy","Onset"], digits=4))

        results[name] = {"roc_auc":roc, "pr_auc":pr, "best_f2":b_f2,
                         "best_threshold":b_thr, "probs":probs}
        if b_f2 > best_f2:
            best_f2 = b_f2
            best_name = name

    print(f"\n>> BEST: {best_name} — F2={results[best_name]['best_f2']:.4f}, "
          f"ROC={results[best_name]['roc_auc']:.4f}, "
          f"PR={results[best_name]['pr_auc']:.4f}")
    return results, best_name


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════

def fairness_audit(y_va, probs, thr, demo):
    banner("PHASE 6: FAIRNESS AUDIT")
    p = (probs > thr).astype(int)
    d = demo.copy()
    d["yt"] = y_va.values
    d["yp"] = p
    d["prob"] = probs

    def check(col, labels=None):
        print(f"\n  By {col}:")
        print(f"  {'Group':<15} {'N':>7} {'ROC':>7} {'F2':>6} {'FPR':>6} {'FNR':>6}")
        for v in sorted(d[col].dropna().unique()):
            m = d[col] == v
            sy = d.loc[m,"yt"]
            sp = d.loc[m,"yp"]
            sprob = d.loc[m,"prob"]
            if sy.nunique()<2 or len(sy)<50: continue
            r = roc_auc_score(sy, sprob)
            f = fbeta_score(sy, sp, beta=2)
            tn,fp,fn,tp = confusion_matrix(sy,sp).ravel()
            fpr = fp/max(fp+tn,1)
            fnr = fn/max(fn+tp,1)
            lbl = labels.get(v,str(v)) if labels else str(v)
            print(f"  {lbl:<15} {len(sy):>7,} {r:>7.4f} {f:>6.4f} {fpr:>6.4f} {fnr:>6.4f}")

    if "RAGENDER" in d.columns:
        check("RAGENDER", {1:"Male", 2:"Female"})
    if "RARACEM" in d.columns:
        check("RARACEM", {1:"White", 2:"Black", 3:"Other"})
    if "Age" in d.columns:
        d["AB"] = pd.cut(d["Age"], bins=[0,55,65,75,85,120],
                         labels=["50-55","56-65","66-75","76-85","85+"])
        check("AB")
    print("\n  >> Fairness audit complete")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 7: EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════

def explain(model, X_va, features, name="LightGBM"):
    banner("PHASE 7: EXPLAINABILITY (SHAP)")
    if not HAS_SHAP:
        print("[SKIP] shap not installed")
        return

    print(f"SHAP for {name}...")
    sample = X_va.sample(n=min(2000, len(X_va)), random_state=SEED)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample)
        if isinstance(sv, list): sv = sv[1]

        plt.figure(figsize=(12,10))
        shap.summary_plot(sv, sample, max_display=30, show=False, plot_size=(12,10))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/"shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  >> {OUTPUT_DIR/'shap_summary.png'}")

        plt.figure(figsize=(12,10))
        shap.summary_plot(sv, sample, plot_type="bar", max_display=30,
                          show=False, plot_size=(12,10))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/"shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  >> {OUTPUT_DIR/'shap_importance.png'}")

        hi = np.argmax(sv.sum(axis=1))
        plt.figure(figsize=(12,8))
        ev = explainer.expected_value
        if not np.isscalar(ev): ev = ev[1]
        shap.waterfall_plot(shap.Explanation(
            values=sv[hi], base_values=ev,
            data=sample.iloc[hi], feature_names=list(sample.columns)
        ), max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/"shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  >> {OUTPUT_DIR/'shap_waterfall.png'}")
    except Exception as e:
        print(f"  [WARN] SHAP failed: {e}")


def export_importance(models, features):
    print("\n--- Feature Importance ---")
    dfs = []
    for n, m in models.items():
        if n == "Ensemble_Meta": continue
        if hasattr(m, "feature_importances_"):
            dfs.append(pd.DataFrame({"Feature":features, f"{n}":m.feature_importances_}))
    if not dfs: return None
    mg = dfs[0]
    for d in dfs[1:]: mg = pd.merge(mg, d, on="Feature", how="outer")
    icols = [c for c in mg.columns if c != "Feature"]
    mg["Avg"] = mg[icols].mean(axis=1)
    mg.sort_values("Avg", ascending=False, inplace=True)
    print(mg.head(30).to_string(index=False))
    mg.to_csv(OUTPUT_DIR/"feature_importance.csv", index=False)

    plt.figure(figsize=(12,10))
    top = mg.head(30)
    plt.barh(top["Feature"].values[::-1], top["Avg"].values[::-1], color="#2196F3")
    plt.title("Top 30 Predictors of Chronic Disease Onset", fontsize=14)
    plt.xlabel("Average Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    return mg


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    banner("HEA HACKATHON — DISEASE PREDICTION PIPELINE v2")
    import gc

    df = load_and_reshape(DATA_PATH)
    df = engineer_features(df)
    gc.collect()

    features, X_tr, y_tr, X_va, y_va, demo = prepare_splits(df)
    del df; gc.collect()

    models, predictions = train_models(X_tr, y_tr, X_va, y_va, features)
    results, best = evaluate(y_va, predictions)
    fairness_audit(y_va, results[best]["probs"], results[best]["best_threshold"], demo)

    if "LightGBM" in models:
        explain(models["LightGBM"], X_va, features, "LightGBM")
    export_importance(models, features)

    banner("FINAL SUMMARY")
    print(f"{'Model':<15} {'ROC':>7} {'PR':>7} {'F2':>7} {'Thr':>6}")
    print("-"*45)
    for n, r in results.items():
        print(f"{n:<15} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f} "
              f"{r['best_f2']:>7.4f} {r['best_threshold']:>6.3f}")
    print(f"\n>> BEST: {best} — "
          f"ROC={results[best]['roc_auc']:.4f} "
          f"PR={results[best]['pr_auc']:.4f} "
          f"F2={results[best]['best_f2']:.4f}")

    rj = {n:{k:float(v) for k,v in r.items() if k!="probs"} for n,r in results.items()}
    with open(OUTPUT_DIR/"results.json","w") as f: json.dump(rj, f, indent=2)
    print(f"\n>> All outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
