"""
Quick investigation: what does the original notebook target look like vs ours?
Also check if ConditionCount (CONDE) correlates with our target (leakage).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "randhrs1992_2022v1.sas7bdat"
OUT = Path(__file__).parent / "output" / "investigation.txt"

lines = []

print("Loading data...")
df_raw = pd.read_sas(str(DATA_PATH))
print(f"Shape: {df_raw.shape}")

# ---------- Check original notebook approach ----------
# The original notebook used just "DIAB" (not "DIABE") etc.
# Let's compare both forms for a few waves

lines.append("="*60)
lines.append("COMPARISON: 'Ever' vs 'Current' disease flags")
lines.append("="*60)
for w in [5, 8, 10, 12]:
    lines.append(f"\n--- Wave {w} ---")
    for short, ever in [("DIAB","DIABE"),("HEART","HEARTE"),("STROK","STROKE"),
                        ("LUNG","LUNGE"),("HIBP","HIBPE")]:
        cs = f"R{w}{short}"
        ce = f"R{w}{ever}"
        if cs in df_raw.columns and ce in df_raw.columns:
            vs = df_raw[cs].dropna()
            ve = df_raw[ce].dropna()
            lines.append(f"  {short:6} mean={vs.mean():.4f} n={len(vs):>6} | "
                        f"{ever:8} mean={ve.mean():.4f} n={len(ve):>6} | "
                        f"diff={ve.mean()-vs.mean():.4f}")

# ---------- Check CONDE correlation with disease flags ----------
lines.append("\n" + "="*60)
lines.append("CONDE (condition count) vs disease presence")
lines.append("="*60)
for w in [8, 10, 12]:
    conde = f"R{w}CONDE"
    if conde in df_raw.columns:
        cvals = df_raw[conde].dropna()
        lines.append(f"\n  Wave {w} CONDE: mean={cvals.mean():.2f}, "
                     f"max={cvals.max():.0f}, min={cvals.min():.0f}")
        # CONDE = sum of diabetes, heart, stroke, lung, cancer, arthritis, hibp, psych
        # This DIRECTLY counts the diseases we're trying to predict!
        # So using CONDE as a feature would be massive leakage

# ---------- Check "onset" definition consistency ----------
lines.append("\n" + "="*60)
lines.append("TARGET DEFINITION ANALYSIS")
lines.append("="*60)

# Build the same target as solution.py
WAVES = list(range(3, 17))
tgt_suffixes = {"T_Diab":"DIABE","T_Heart":"HEARTE","T_Stroke":"STROKE",
                "T_Lung":"LUNGE","T_Cancer":"CANCRE","T_Arthritis":"ARTHRE"}

person_waves = []
for w in WAVES:
    pw = {"HHIDPN": df_raw["HHIDPN"].values, "Wave": w}
    for nm, sf in tgt_suffixes.items():
        col = f"R{w}{sf}"
        pw[nm] = df_raw[col].values if col in df_raw.columns else np.nan
    person_waves.append(pd.DataFrame(pw))

df = pd.concat(person_waves, ignore_index=True)
df = df.sort_values(["HHIDPN","Wave"])
for c in tgt_suffixes.keys():
    df[c] = df.groupby("HHIDPN")[c].ffill()
    df[c] = df[c].fillna(0).clip(0,1)

df["DiseaseCount"] = df[list(tgt_suffixes.keys())].sum(axis=1)
df["IsSick"] = (df["DiseaseCount"]>0).astype(int)
df["NextSick"] = df.groupby("HHIDPN")["IsSick"].shift(-1)
df["Y"] = 0
df.loc[(df["IsSick"]==0) & (df["NextSick"]==1), "Y"] = 1

at_risk = df[(df["IsSick"]==0) & (df["NextSick"].notna())]
lines.append(f"\nAt-risk cohort: {len(at_risk):,}")
lines.append(f"Events: {at_risk['Y'].sum():,} ({at_risk['Y'].mean():.2%})")

# Per wave event rates
lines.append(f"\nPer-wave event rates:")
for w in WAVES:
    mask = at_risk["Wave"]==w
    if mask.sum() > 0:
        rate = at_risk.loc[mask, "Y"].mean()
        n = mask.sum()
        lines.append(f"  Wave {w}: {rate:.2%} (n={n:,})")

lines.append(f"\n** NOTE: If event rate varies significantly by wave,")
lines.append(f"   then WaveNum was capturing this temporal trend as the")
lines.append(f"   dominant signal — NOT a health signal.")

# ---------- What about using CURRENT-WAVE disease flag (short form)? ----------
lines.append("\n" + "="*60)
lines.append("ALTERNATIVE TARGET: Using short form (DIAB vs DIABE)")
lines.append("="*60)

tgt_short = {"T_Diab":"DIAB","T_Heart":"HEART","T_Stroke":"STROK",
             "T_Lung":"LUNG","T_Cancer":"CANCRE","T_Arth":"ARTHR"}

person_waves2 = []
for w in WAVES:
    pw = {"HHIDPN": df_raw["HHIDPN"].values, "Wave": w}
    for nm, sf in tgt_short.items():
        col = f"R{w}{sf}"
        pw[nm] = df_raw[col].values if col in df_raw.columns else np.nan
    person_waves2.append(pd.DataFrame(pw))

df2 = pd.concat(person_waves2, ignore_index=True)
df2 = df2.sort_values(["HHIDPN","Wave"])
for c in tgt_short.keys():
    df2[c] = df2.groupby("HHIDPN")[c].ffill()
    df2[c] = df2[c].fillna(0).clip(0,1)

df2["DiseaseCount"] = df2[list(tgt_short.keys())].sum(axis=1)
df2["IsSick"] = (df2["DiseaseCount"]>0).astype(int)
df2["NextSick"] = df2.groupby("HHIDPN")["IsSick"].shift(-1)
df2["Y"] = 0
df2.loc[(df2["IsSick"]==0) & (df2["NextSick"]==1), "Y"] = 1

at_risk2 = df2[(df2["IsSick"]==0) & (df2["NextSick"].notna())]
lines.append(f"\nShort-form at-risk cohort: {len(at_risk2):,}")
lines.append(f"Events: {at_risk2['Y'].sum():,} ({at_risk2['Y'].mean():.2%})")

lines.append(f"\nPer-wave event rates (short form):")
for w in WAVES:
    mask = at_risk2["Wave"]==w
    if mask.sum() > 0:
        rate = at_risk2.loc[mask, "Y"].mean()
        n = mask.sum()
        lines.append(f"  Wave {w}: {rate:.2%} (n={n:,})")

# ---------- Check HighBP (HIBPE) as potential leakage ----------
lines.append("\n" + "="*60)
lines.append("HIBPE as feature — is it leakage?")
lines.append("="*60)
lines.append("HIBPE is 'ever diagnosed with high blood pressure'")
lines.append("It's NOT in the CONDE disease list we use as target")
lines.append("So it's safe as a predictor — it's a RISK FACTOR")

result = "\n".join(lines)
with open(str(OUT), "w", encoding="utf-8") as f:
    f.write(result)
print(f"Written to {OUT}")
