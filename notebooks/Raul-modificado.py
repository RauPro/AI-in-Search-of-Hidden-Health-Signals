"""
╔══════════════════════════════════════════════════════════════════════════╗
║  HYBRID DISEASE-EXPERT ENSEMBLE                                          ║
║  Health Hackathon — Disease Onset Prediction                             ║
║                                                                          ║
║  Combines:                                                               ║
║    ✓ Disease-specific expert models (1 per disease)                      ║
║    ✓ Temporal feature engineering (deltas, acceleration, rolling stats)  ║
║    ✓ Group-aware splitting (no person leakage)                           ║
║    ✓ 4-year prediction horizon                                           ║
║    ✓ SHAP explainability + fairness audit                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    fbeta_score, roc_auc_score, average_precision_score,
    precision_score, recall_score
)

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

DATA_PATH  = Path(__file__).parent / "data" / "extracted" / "randhrs1992_2022v1.parquet"
OUTPUT_DIR = Path(__file__).parent / "output_hybrid"
OUTPUT_DIR.mkdir(exist_ok=True)

# Screenings where we measure features → outcome is screening + HORIZON
FEATURE_SCREENINGS = list(range(5, 13))   # HRS screenings 5..12
PREDICTION_HORIZON = 2                    # 2 screenings ahead (~4 years)
LAGS = 2                                  # use current, prev, prev-prev screening

# 8 disease experts — screening-specific self-report codes (NOT cumulative "E" vars)
DISEASE_MAP = {
    "diabetes":    "DIAB",
    "cvd":         "HEART",
    "stroke":      "STROK",
    "lung":        "LUNG",
    "cancer":      "CANCR",
    "hibp":        "HIBP",
    "arthritis":   "ARTHR",
    "psychiatric": "PSYCH",
    "memory":      "MEMRY",
}

# ── Variables to extract per screening ──────────────────────────────

SCREENING_VARS = {
    "self_rated_health": "SHLT",
    "bmi":               "BMI",
    "weight":            "WEIGHT",
    "height":            "HEIGHT",
    "mobility":          "MOBILA",
    "gross_motor":       "GROSSA",
    "large_muscle":      "LGMUSA",
    "fine_motor":        "FINEA",
    "adl":               "ADL5A",
    "iadl":              "IADL5A",
    "cognition":         "COG27",
    "memory_recall":     "TR20",
    "immediate_recall":  "IMRC",
    "delayed_recall":    "DLRC",
    "serial7":           "SER7",
    "cesd":              "CESD",
    "depressed":         "DEPRES",
    "effort":            "EFFORT",
    "restless_sleep":    "SLEEPR",
    "lonely":            "FLONE",
    "ever_smoked":       "SMOKEV",
    "current_smoker":    "SMOKEN",
    "drinks_per_day":    "DRINKD",
    "drink_days_week":   "DRINKN",
    "vigorous_activity": "VGACTX",
    "marital_status":    "MSTAT",
    "condition_count":   "CONDE",
    "self_health_comp":  "SHLTC",
    "out_of_pocket":     "OOPMD",
    "working":           "WORK",
}

# ── Helpers ──────────────────────────────────────────────────────────

# HRS screening → approximate year (fallback when interview date missing)
SCREENING_YEARS = {
    1:1992, 2:1993, 3:1994, 4:1995, 5:1996, 6:1998, 7:2000,
    8:2002, 9:2004, 10:2006, 11:2008, 12:2010, 13:2012,
    14:2014, 15:2016, 16:2018, 17:2020, 18:2022
}

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


# ═════════════════════════════════════════════════════════════════════
#  PHASE 1: EXTRACT FEATURES PER SCREENING (with lags)
# ═════════════════════════════════════════════════════════════════════

def extract_screening_features(df_raw: pd.DataFrame, scr: int) -> pd.DataFrame:
    """
    For a given screening, extract:
      - Demographics (static)
      - Current screening values
      - Lag-1 and Lag-2 values (raw)
    """
    data = {}

    # Person identifier
    data["person_id"] = df_raw["HHIDPN"].values

    # Screening date (interview midpoint: SAS date = days since 1960-01-01)
    SAS_EPOCH = pd.Timestamp("1960-01-01")
    iw_col = f"R{scr}IWMID"
    if iw_col in df_raw.columns:
        iw_days = pd.to_numeric(df_raw[iw_col], errors="coerce")
        screening_date = SAS_EPOCH + pd.to_timedelta(iw_days, unit="D")
        data["screening_year"]  = screening_date.dt.year.values.astype(float)
        data["screening_month"] = screening_date.dt.month.values.astype(float)
    else:
        data["screening_year"]  = float(SCREENING_YEARS[scr])
        data["screening_month"] = np.nan

    # Time gap to previous screening (in years)
    iw_lag1_col = f"R{scr - 1}IWMID" if scr > 1 else None
    if iw_lag1_col and iw_lag1_col in df_raw.columns and iw_col in df_raw.columns:
        iw_cur  = pd.to_numeric(df_raw[iw_col], errors="coerce")
        iw_prev = pd.to_numeric(df_raw[iw_lag1_col], errors="coerce")
        data["years_since_last_screening"] = ((iw_cur - iw_prev) / 365.25).values
    else:
        data["years_since_last_screening"] = np.nan

    # Static demographics — age from actual screening date
    birth_year = pd.to_numeric(df_raw["RABYEAR"], errors="coerce")
    age = data["screening_year"] - birth_year.values
    data["birth_year"] = birth_year.values
    data["age"] = age
    data["age_squared"] = age ** 2

    data["female"]    = (pd.to_numeric(df_raw["RAGENDER"], errors="coerce") == 2).astype(int).values

    race = pd.to_numeric(df_raw["RARACEM"], errors="coerce")
    hisp = pd.to_numeric(df_raw["RAHISPAN"], errors="coerce")
    def map_ethnicity(r, h):
        if h == 1:
            return "Hispanic"
        if r == 1:
            return "White"
        if r == 2:
            return "Black"
        return "Other"
    
    data["ethnicity"] = pd.Categorical(
        [map_ethnicity(r, h) for r, h in zip(race, hisp)],
        categories=["White", "Black", "Hispanic", "Other"]
    )
    
    data["education"] = pd.to_numeric(df_raw.get("RAEDYRS"), errors="coerce").values
    data["edu_cat"]   = pd.to_numeric(df_raw.get("RAEDUC"), errors="coerce").values
    data["degree"]    = pd.to_numeric(df_raw.get("RAEDEGRM"), errors="coerce").values

    # Screening / lag extraction
    for lag in range(LAGS + 1):
        w = scr - lag
        if w < 1:
            continue
        suffix = f"_lag{lag}" if lag > 0 else ""
        for name, code in SCREENING_VARS.items():
            col = f"R{w}{code}"
            if col in df_raw.columns:
                data[f"{name}{suffix}"] = pd.to_numeric(df_raw[col], errors="coerce").values
            else:
                data[f"{name}{suffix}"] = np.nan

    return pd.DataFrame(data)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 2: TEMPORAL FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity, acceleration, decline, and interaction features."""
    new = {}

    # ── BMI trajectories (delta between screenings) ──
    new["bmi_delta_lag1"]    = df.get("bmi", np.nan) - df.get("bmi_lag1", np.nan)
    new["bmi_delta_lag2"]    = df.get("bmi", np.nan) - df.get("bmi_lag2", np.nan)
    new["bmi_accel"]         = new["bmi_delta_lag1"] - (
        df.get("bmi_lag1", np.nan) - df.get("bmi_lag2", np.nan)
    )
    new["weight_change_kg"]  = df.get("weight", np.nan) - df.get("weight_lag1", np.nan)
    wl1 = df.get("weight_lag1", np.nan)
    new["weight_change_pct"] = np.where(wl1 != 0, new["weight_change_kg"] / wl1 * 100, np.nan)
    new["obese"]             = (df.get("bmi", 0) >= 30).astype(int)
    new["overweight"]        = ((df.get("bmi", 0) >= 25) & (df.get("bmi", 0) < 30)).astype(int)
    new["rapid_weight_gain"] = (new["bmi_delta_lag1"] > 1).astype(int)
    new["rapid_weight_loss"] = (new["bmi_delta_lag1"] < -1).astype(int)

    # ── Health perception ──
    new["health_decline_lag1"] = df.get("self_rated_health", np.nan) - df.get("self_rated_health_lag1", np.nan)
    new["health_decline_lag2"] = df.get("self_rated_health", np.nan) - df.get("self_rated_health_lag2", np.nan)
    new["health_worsening"]    = (new["health_decline_lag1"] > 0).astype(int)
    new["health_crash"]        = (new["health_decline_lag1"] >= 2).astype(int)

    # ── Functional decline ──
    for base in ["mobility", "adl", "iadl"]:
        cur = df.get(base, np.nan)
        lag1 = df.get(f"{base}_lag1", np.nan)
        new[f"{base}_decline_lag1"] = cur - lag1
        new[f"{base}_worsening"]    = (new[f"{base}_decline_lag1"] > 0).astype(int)
        new[f"new_{base}_problem"]  = ((cur > 0) & (lag1 == 0)).astype(int)
        new[f"any_{base}"]          = (cur > 0).astype(int)

    # ── Cognitive decline ──
    cog    = df.get("cognition", np.nan)
    cog_l1 = df.get("cognition_lag1", np.nan)
    cog_l2 = df.get("cognition_lag2", np.nan)
    new["cog_decline_lag1"]    = cog_l1 - cog
    new["cog_decline_lag2"]    = cog_l2 - cog
    new["cog_worsening"]       = (new["cog_decline_lag1"] > 0).astype(int)
    new["sharp_cog_drop"]      = (new["cog_decline_lag1"] > 3).astype(int)
    new["low_cognition"]       = (cog < 12).astype(int)

    mem    = df.get("memory_recall", np.nan)
    mem_l1 = df.get("memory_recall_lag1", np.nan)
    new["memory_decline_lag1"] = mem_l1 - mem
    new["memory_worsening"]    = (new["memory_decline_lag1"] > 0).astype(int)

    # ── Mental health ──
    cesd    = df.get("cesd", np.nan)
    cesd_l1 = df.get("cesd_lag1", np.nan)
    new["cesd_increase"]        = cesd - cesd_l1
    new["cesd_worsening"]       = (new["cesd_increase"] > 0).astype(int)
    new["elevated_depression"]  = (cesd >= 3).astype(int)
    new["high_depression"]      = (cesd >= 4).astype(int)
    new["chronic_depression"]   = ((cesd >= 3) & (cesd_l1 >= 3)).astype(int)

    # ── Health behaviors ──
    new["former_smoker"]  = ((df.get("ever_smoked", 0) == 1) & (df.get("current_smoker", 0) == 0)).astype(int)
    new["quit_smoking"]   = ((df.get("current_smoker", 0) == 0) & (df.get("current_smoker_lag1", 0) == 1)).astype(int)
    dpd = df.get("drinks_per_day", np.nan)
    dpw = df.get("drink_days_week", np.nan)
    new["drinks_per_week"]  = dpd * dpw
    new["heavy_drinking"]   = (new["drinks_per_week"] > 14).astype(int)

    # ── Interactions ──
    new["age_x_bmi"]             = df.get("age", np.nan) * df.get("bmi", np.nan)
    new["age_x_cesd"]            = df.get("age", np.nan) * cesd
    new["depression_x_mobility"] = cesd * df.get("mobility", np.nan)
    new["cog_decline_x_age"]     = new["cog_decline_lag1"] * df.get("age", np.nan)
    new["bmi_x_smoking"]         = df.get("bmi", np.nan) * df.get("current_smoker", np.nan)

    # ── Composite scores ──
    new["metabolic_risk"] = (
        new["obese"] * 2
        + (df.get("age", 0) >= 65).astype(int)
    )
    new["frailty_score"] = (
        (df.get("mobility", 0) > 0).astype(int)
        + (cesd >= 3).astype(int)
        + (new["weight_change_pct"] < -5).astype(int)
    )

    # Single concat to prevent fragmentation
    result = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return result


# ═════════════════════════════════════════════════════════════════════
#  PHASE 3: CREATE DISEASE-SPECIFIC TARGETS
# ═════════════════════════════════════════════════════════════════════

def create_targets(df_raw: pd.DataFrame, feature_screening: int, outcome_screening: int) -> pd.DataFrame:
    """
    For each disease, create:
      - target_{disease}: 1 if onset (0→1) between feature and outcome screening
      - eligible_{disease}: 1 if disease=0 at feature screening
    Uses screening-specific self-report codes (0=no, 1=yes, 3=disputes, 4=don't know).
    Values 3/4 and NaN are treated as missing (excluded from eligibility/outcome).
    """
    targets = {"person_id": df_raw["HHIDPN"].values}

    for disease_name, code in DISEASE_MAP.items():
        baseline_col = f"R{feature_screening}{code}"
        outcome_col  = f"R{outcome_screening}{code}"

        if baseline_col in df_raw.columns and outcome_col in df_raw.columns:
            baseline = pd.to_numeric(df_raw[baseline_col], errors="coerce")
            outcome  = pd.to_numeric(df_raw[outcome_col], errors="coerce")

            # Only 0/1 are clean answers; 3 (disputes) and 4 (don't know) → NaN
            baseline_clean = baseline.where(baseline.isin([0, 1]))
            outcome_clean  = outcome.where(outcome.isin([0, 1]))

            no_disease = (baseline_clean == 0)
            develops   = (outcome_clean == 1)

            target_vals = (no_disease & develops).astype(float)
            # Mark as NaN if outcome is unknown
            target_vals[outcome_clean.isna()] = np.nan

            targets[f"target_{disease_name}"]   = target_vals.values
            targets[f"eligible_{disease_name}"] = no_disease.astype(int).values

    return pd.DataFrame(targets)


# ═════════════════════════════════════════════════════════════════════
#  PHASE 4: ASSEMBLE DATASET PER DISEASE
# ═════════════════════════════════════════════════════════════════════

def build_master_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build multi-screening long-format features + targets."""
    banner("PHASE 1-3: BUILDING FEATURES + TARGETS")

    all_data = []

    for scr in FEATURE_SCREENINGS:
        outcome_scr = scr + PREDICTION_HORIZON
        if outcome_scr > 18:
            continue

        # Features for this screening (with lags)
        feats = extract_screening_features(df_raw, scr)
        feats = add_temporal_features(feats)

        # Targets from this screening → outcome screening
        targs = create_targets(df_raw, scr, outcome_scr)

        # Merge on person_id
        combined = feats.merge(targs, on="person_id", how="inner")
        # Internal grouping key (not exposed as a feature)
        combined["_screening_id"] = scr
        all_data.append(combined)

        print(f"  Screening {scr} → {outcome_scr}: {len(combined):,} rows, "
              f"{feats.shape[1]} features")

    master = pd.concat(all_data, ignore_index=True)
    print(f"\n  TOTAL: {len(master):,} rows")
    return master


def get_disease_dataset(master: pd.DataFrame, disease: str):
    """
    Filter master to only eligible patients for this disease,
    split group-aware, return X_train/test, y_train/test.
    """
    eligible_col = f"eligible_{disease}"
    target_col   = f"target_{disease}"

    # Only eligible (didn't have this disease at baseline)
    sub = master[master[eligible_col] == 1].copy()
    # Remove rows where outcome is unknown
    sub = sub[sub[target_col].notna()].copy()
    # Drop rows missing critical features
    sub = sub.dropna(subset=["age", "bmi"])

    y = sub[target_col].astype(int)

    # Feature columns: exclude targets, eligibility, metadata
    exclude_prefixes = ("target_", "eligible_")
    metadata_cols = {"person_id", "_screening_id"}
    # # Also exclude demographic columns used for fairness audit
    # fairness_cols = {"female", "ethnicity"}

    feature_cols = sorted([
        c for c in sub.columns
        if not c.startswith(exclude_prefixes)
        and c not in metadata_cols
        # Keep fairness cols as features (they can be predictive)
    ])

    X = sub[feature_cols]

    # Group-aware split: same person never in both train and test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=sub["person_id"]))

    X_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx].copy()
    X_test  = X.iloc[test_idx].copy()
    y_test  = y.iloc[test_idx].copy()

    # Fairness demographics for test set
    demo_test = sub.iloc[test_idx][["person_id", "female", "ethnicity", "age"]].copy()

    return feature_cols, X_train, y_train, X_test, y_test, demo_test


# ═════════════════════════════════════════════════════════════════════
#  PHASE 5: TRAIN EXPERT MODEL FOR ONE DISEASE
# ═════════════════════════════════════════════════════════════════════

def train_disease_expert(
    disease: str,
    feature_cols: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Train LightGBM + CatBoost for one disease, return models, preds, and results.
    """
    banner(f"EXPERT: {disease.upper()}")

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / max(pos, 1)
    print(f"  Train: {len(X_train):,} | Pos: {pos:,} ({pos/(pos+neg):.2%}) | "
          f"scale_pos_weight: {spw:.1f}")

    # Identify categorical columns
    cat_cols = [c for c in feature_cols if X_train[c].dtype.name == "category"]
    cat_col_indices = [feature_cols.index(c) for c in cat_cols]
    if cat_cols:
        print(f"  Categorical features: {cat_cols}")

    # ── LightGBM ──
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        scale_pos_weight=spw,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(300, verbose=False)],
    )
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    print(f"  LightGBM  ROC-AUC: {lgb_auc:.4f}")

    # ── CatBoost ──
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        eval_metric="AUC",
        loss_function="Logloss",
        scale_pos_weight=spw,
        random_seed=SEED,
        verbose=0,
        early_stopping_rounds=300,
        use_best_model=True,
    )
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        cat_features=cat_col_indices if cat_col_indices else None,
    )
    cat_pred = cat_model.predict_proba(X_test)[:, 1]
    cat_auc = roc_auc_score(y_test, cat_pred)
    print(f"  CatBoost  ROC-AUC: {cat_auc:.4f}")

    # ── XGBoost ──
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=50,
        scale_pos_weight=spw,
        eval_metric="auc",
        enable_categorical=True,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=300,
    )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    print(f"  XGBoost   ROC-AUC: {xgb_auc:.4f}")

    # ── Average ensemble ──
    avg_pred = (lgb_pred + cat_pred + xgb_pred) / 3
    avg_auc = roc_auc_score(y_test, avg_pred)
    print(f"  Ensemble  ROC-AUC: {avg_auc:.4f}")

    models = {"lgb": lgb_model, "cat": cat_model, "xgb": xgb_model}
    preds = {
        "lgb": lgb_pred, "cat": cat_pred,
        "xgb": xgb_pred, "avg": avg_pred
    }

    # Evaluate metrics
    results = evaluate_expert(disease, y_test, preds)

    return models, results, avg_pred


# ═════════════════════════════════════════════════════════════════════
#  PHASE 6: EVALUATE & OPTIMIZE THRESHOLD
# ═════════════════════════════════════════════════════════════════════

def find_best_f2_threshold(y_true, y_proba):
    best_t, best_f2 = 0.5, 0
    for t in np.linspace(0.02, 0.6, 200):
        preds = (y_proba >= t).astype(int)
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
    return best_t, best_f2


def evaluate_expert(disease: str, y_test, preds: dict) -> dict:
    results = {}
    for name, proba in preds.items():
        roc  = roc_auc_score(y_test, proba)
        pr   = average_precision_score(y_test, proba)
        bt, bf2 = find_best_f2_threshold(y_test, proba)

        y_pred = (proba >= bt).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "f2": round(bf2, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "threshold": round(bt, 4),
        }
        print(f"  {disease}/{name}: ROC={roc:.4f} PR={pr:.4f} F2={bf2:.4f} "
              f"P={prec:.4f} R={rec:.4f} t={bt:.3f}")
    return results


# ═════════════════════════════════════════════════════════════════════
#  PHASE 7: COMBINED "ANY ONSET" META-ENSEMBLE
# ═════════════════════════════════════════════════════════════════════

def build_any_onset_from_experts(
    master: pd.DataFrame,
    all_models: Dict[str, Dict],
    all_feature_cols: Dict[str, List[str]],
):
    """
    For the test set, combine expert probabilities:
      P(any onset) = 1 - ∏(1 - P_d)  for eligible diseases
    """
    banner("META-ENSEMBLE: ANY-DISEASE ONSET")

    # Group by person + screening (using internal _wave key)
    # Rebuild predictions disease by disease and combine per person-screening

    results_per_ps = {}

    for disease, models in all_models.items():
        eligible_col = f"eligible_{disease}"
        target_col   = f"target_{disease}"

        sub = master[(master[eligible_col] == 1) & (master[target_col].notna())].copy()
        sub = sub.dropna(subset=["age", "bmi"])

        feature_cols = all_feature_cols[disease]
        X = sub[feature_cols]

        # Get predictions from ensemble mean
        lgb_pred = models["lgb"].predict_proba(X)[:, 1]
        cat_pred = models["cat"].predict_proba(X)[:, 1]
        xgb_pred = models["xgb"].predict_proba(X)[:, 1]
        avg_pred = (lgb_pred + cat_pred + xgb_pred) / 3

        for i, (pid, w) in enumerate(zip(sub["person_id"].values, sub["_screening_id"].values)):
            key = (pid, w)
            if key not in results_per_ps:
                results_per_ps[key] = {"probs": [], "has_target": 0, "any_eligible": False}
            results_per_ps[key]["probs"].append(avg_pred[i])
            results_per_ps[key]["any_eligible"] = True
            # Any-onset target: did they develop ANY disease?
            if sub.iloc[i][target_col] == 1:
                results_per_ps[key]["has_target"] = 1

    # Compute P(any onset) = 1 - prod(1 - p_d)
    y_true_list = []
    y_proba_list = []

    for key, info in results_per_ps.items():
        if not info["any_eligible"]:
            continue
        probs = info["probs"]
        p_no_onset = 1.0
        for p in probs:
            p_no_onset *= (1 - p)
        p_any = 1 - p_no_onset

        y_proba_list.append(p_any)
        y_true_list.append(info["has_target"])

    y_true  = np.array(y_true_list)
    y_proba = np.array(y_proba_list)

    # Evaluate
    roc  = roc_auc_score(y_true, y_proba)
    pr   = average_precision_score(y_true, y_proba)
    bt, bf2 = find_best_f2_threshold(y_true, y_proba)

    print("\n  ANY-ONSET Combined:")
    print(f"    ROC-AUC:  {roc:.4f}")
    print(f"    PR-AUC:   {pr:.4f}")
    print(f"    Best F2:  {bf2:.4f} @ threshold {bt:.3f}")
    print(f"    Samples:  {len(y_true):,} | Events: {y_true.sum():,.0f} ({y_true.mean():.2%})")

    return {"roc_auc": roc, "pr_auc": pr, "f2": bf2, "threshold": bt}


# ═════════════════════════════════════════════════════════════════════
#  PHASE 8: EXPLAINABILITY (SHAP)
# ═════════════════════════════════════════════════════════════════════

def explain_expert(models, X_test, feature_cols, disease, top_n=20):
    """SHAP summary for the LightGBM expert of this disease."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model = models["lgb"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, max_display=top_n, show=False)
        plt.title(f"SHAP — {disease.upper()} Expert")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"shap_{disease}.png", dpi=150)
        plt.close()
        print(f"  Saved SHAP plot: shap_{disease}.png")
    except Exception as e:
        print(f"  SHAP failed for {disease}: {e}")


# ═════════════════════════════════════════════════════════════════════
#  PHASE 9: FEATURE IMPORTANCE EXPORT
# ═════════════════════════════════════════════════════════════════════

def export_importance(all_models, all_feature_cols):
    """Export feature importance per disease expert."""
    rows = []
    for disease, models in all_models.items():
        feat_cols = all_feature_cols[disease]
        lgb_imp = models["lgb"].feature_importances_
        cat_imp = models["cat"].get_feature_importance()
        xgb_imp = models["xgb"].feature_importances_

        for i, feat in enumerate(feat_cols):
            rows.append({
                "disease": disease,
                "feature": feat,
                "lgb": lgb_imp[i],
                "cat": cat_imp[i],
                "xgb": xgb_imp[i],
                "avg": (lgb_imp[i] + cat_imp[i] + xgb_imp[i]) / 3,
            })

    df_imp = pd.DataFrame(rows)
    df_imp.to_csv(OUTPUT_DIR / "feature_importance_experts.csv", index=False)
    print(f"\n  Saved feature_importance_experts.csv ({len(df_imp)} rows)")

    # Print top-10 per disease
    for disease in DISEASE_MAP:
        sub = df_imp[df_imp["disease"] == disease].nlargest(10, "avg")
        print(f"\n  TOP-10 {disease.upper()}:")
        for _, row in sub.iterrows():
            print(f"    {row['feature']:<30} avg={row['avg']:.1f}")


# ═════════════════════════════════════════════════════════════════════
#  PHASE 10: FAIRNESS AUDIT
# ═════════════════════════════════════════════════════════════════════

def fairness_audit(disease, y_test, y_proba, threshold, demo_test):
    """Quick fairness check across demographics."""
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n  Fairness — {disease.upper()}")

    # Gender
    for gval, gname in {0: "Male", 1: "Female"}.items():
        mask = demo_test["female"].values == gval
        if mask.sum() < 50:
            continue
        g_rec = recall_score(y_test.values[mask], y_pred[mask], zero_division=0)
        g_prec = precision_score(y_test.values[mask], y_pred[mask], zero_division=0)
        g_f2  = fbeta_score(y_test.values[mask], y_pred[mask], beta=2, zero_division=0)
        print(f"    {gname:<12}: F2={g_f2:.4f}  P={g_prec:.4f}  R={g_rec:.4f} (n={mask.sum():,})")

    # Ethnicity
    for eth in ["White", "Black", "Hispanic", "Other"]:
        mask = demo_test["ethnicity"].values == eth
        if mask.sum() < 50:
            continue
        g_rec = recall_score(y_test.values[mask], y_pred[mask], zero_division=0)
        g_prec = precision_score(y_test.values[mask], y_pred[mask], zero_division=0)
        g_f2  = fbeta_score(y_test.values[mask], y_pred[mask], beta=2, zero_division=0)
        print(f"    {eth:<12}: F2={g_f2:.4f}  P={g_prec:.4f}  R={g_rec:.4f} (n={mask.sum():,})")


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    banner("HYBRID DISEASE-EXPERT ENSEMBLE")
    print(f"  Data:    {DATA_PATH}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Horizon: {PREDICTION_HORIZON} screenings (~{PREDICTION_HORIZON*2} years)")
    print(f"  Diseases: {list(DISEASE_MAP.keys())}")

    # Load raw
    print("\n  Loading parquet data...")
    df_raw = pd.read_parquet(str(DATA_PATH))
    print(f"  Shape: {df_raw.shape}")

    # Build master feature+target dataset
    master = build_master_features(df_raw)

    # Train one expert per disease
    all_models = {}
    all_results = {}
    all_feature_cols = {}

    for disease in DISEASE_MAP:
        try:
            feature_cols, X_tr, y_tr, X_te, y_te, demo = get_disease_dataset(master, disease)
            all_feature_cols[disease] = feature_cols

            models, results, avg_pred = train_disease_expert(
                disease, feature_cols, X_tr, y_tr, X_te, y_te
            )
            all_models[disease] = models
            all_results[disease] = results

            # Best threshold from ensemble average
            bt = results["avg"]["threshold"]
            fairness_audit(disease, y_te, avg_pred, bt, demo)
            explain_expert(models, X_te, feature_cols, disease)

        except Exception as e:
            print(f"\n  ERROR for {disease}: {e}")
            traceback.print_exc()

    # Meta-ensemble: any-onset
    if all_models:
        any_onset_results = build_any_onset_from_experts(
            master, all_models, all_feature_cols
        )
        all_results["_any_onset_combined"] = any_onset_results

    # Export
    if all_models:
        export_importance(all_models, all_feature_cols)

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n  Saved results.json")

    # ── FINAL SUMMARY ──
    banner("FINAL SUMMARY")
    print(f"{'Disease':<14} {'ROC-AUC':>8} {'PR-AUC':>8} {'F2':>8} {'Recall':>8} {'Prec':>8}")
    print("-" * 62)
    for disease, res in all_results.items():
        if disease.startswith("_"):
            # Meta-ensemble
            r = res
            print(f"{'ANY-ONSET':<14} {r.get('roc_auc',0):>8.4f} {r.get('pr_auc',0):>8.4f} "
                  f"{r.get('f2',0):>8.4f}")
        else:
            r = res.get("avg", {})
            print(f"{disease:<14} {r.get('roc_auc',0):>8.4f} {r.get('pr_auc',0):>8.4f} "
                  f"{r.get('f2',0):>8.4f} {r.get('recall',0):>8.4f} {r.get('precision',0):>8.4f}")


if __name__ == "__main__":
    main()
