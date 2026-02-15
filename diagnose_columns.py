"""Check Cesar's additional variables."""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "randhrs1992_2022v1.sas7bdat"
OUT = Path(__file__).parent / "output" / "cesar_vars_check.txt"

df = pd.read_sas(str(DATA_PATH))
lines = []

wave = 10
extra_vars = {
    "COG27": "composite cognition",
    "WEIGHT": "weight", "HEIGHT": "height",
    "IMRC": "immediate recall", "DLRC": "delayed recall",
    "SER7": "serial 7s",
    "DEPRES": "felt depressed", "EFFORT": "everything effort",
    "SLEEPR": "restless sleep", "FLONE": "felt lonely",
    "DRINKD": "drinks per day", "DRINKN": "drink days/week",
    "ADL5A": "ADL 5-item", "IADL5A": "IADL 5-item",
    "ADL3A": "ADL 3-item", "IADL3A": "IADL 3-item",
    "ADLA": "ADL sum", "IADLZA": "IADL sum",
    "VIGACT": "vig activity alt",
    "RAEDUC": "education category", "RAEDEGRM": "degree",
    "RABYEAR": "birth year",
    "GROSSA": "gross motor", "STROK": "stroke short",
}

for suffix, desc in extra_vars.items():
    col = f"R{wave}{suffix}" if suffix not in ("RAEDUC","RAEDEGRM","RABYEAR") else suffix
    if col in df.columns:
        nn = df[col].notna().sum()
        pct = nn / len(df) * 100
        lines.append(f"  FOUND  {col:<18} ({desc}): {nn:>6} ({pct:.1f}%)")
    else:
        lines.append(f"  MISS   {col:<18} ({desc})")

# Check wave coverage for found vars
lines.append("\n=== Wave coverage ===")
for suffix in ["COG27","WEIGHT","HEIGHT","IMRC","DLRC","SER7",
               "DEPRES","EFFORT","SLEEPR","FLONE","DRINKD","DRINKN",
               "ADL5A","IADL5A","ADL3A","IADL3A","VIGACT"]:
    found = [w for w in range(1,17) if f"R{w}{suffix}" in df.columns]
    if found:
        lines.append(f"  {suffix:<10}: waves {found}")
    else:
        lines.append(f"  {suffix:<10}: NOT FOUND")

with open(str(OUT), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Written to {OUT}")
