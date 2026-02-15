# Hybrid Expert Models — Disease Onset Prediction Results

> **Dataset**: RAND HRS Longitudinal File (1992–2022 v1)
> **Pipeline**: `hybrid_experts.py` — disease-specific expert models with ensemble averaging
> **Metrics**: ROC-AUC · PR-AUC · F2 Score · Precision · Recall · Threshold

---

## Summary of Results

### Combined Any-Onset Detection

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.6565 |
| **PR-AUC** | 0.4990 |
| **F2 Score** | 0.7318 |
| **Threshold** | 0.60 |

---

## Per-Disease Results

### 🧠 Memory Disease

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.7850 | 0.1339 | 0.3184 | 0.1470 | 0.4494 | 0.0637 |
| CatBoost | 0.8281 | 0.1523 | 0.3321 | 0.1136 | 0.6397 | 0.6000 |
| XGBoost | 0.8314 | 0.1769 | 0.3490 | 0.1405 | 0.5547 | 0.5942 |
| **Ensemble Avg** | **0.8332** | **0.1676** | **0.3612** | **0.1546** | **0.5425** | **0.4543** |

### 🫁 Lung Disease

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.7052 | 0.0813 | 0.2385 | 0.0677 | 0.6460 | 0.0491 |
| CatBoost | 0.7330 | 0.1016 | 0.2618 | 0.0876 | 0.5209 | 0.5563 |
| XGBoost | 0.7306 | 0.1035 | 0.2587 | 0.0970 | 0.4437 | 0.5359 |
| **Ensemble Avg** | **0.7345** | **0.1063** | **0.2680** | **0.1048** | **0.4387** | **0.3989** |

### 🧠 Psychiatric Conditions

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6798 | 0.0989 | 0.2612 | 0.0920 | 0.4837 | 0.0608 |
| CatBoost | 0.7257 | 0.1251 | 0.3065 | 0.1136 | 0.5325 | 0.5446 |
| XGBoost | 0.7300 | 0.1274 | 0.3060 | 0.1018 | 0.6139 | 0.4630 |
| **Ensemble Avg** | **0.7309** | **0.1283** | **0.3044** | **0.1096** | **0.5477** | **0.3610** |

### 🩸 Diabetes

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6705 | 0.1204 | 0.3254 | 0.1026 | 0.7118 | 0.0870 |
| CatBoost | 0.7030 | 0.1426 | 0.3556 | 0.1196 | 0.7013 | 0.4805 |
| XGBoost | 0.7009 | 0.1405 | 0.3512 | 0.1191 | 0.6849 | 0.4747 |
| **Ensemble Avg** | **0.7030** | **0.1420** | **0.3561** | **0.1200** | **0.7006** | **0.3464** |

### ❤️ Cardiovascular Disease (CVD)

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6631 | 0.1610 | 0.3859 | 0.1400 | 0.6879 | 0.1074 |
| CatBoost | 0.6792 | 0.1782 | 0.3995 | 0.1411 | 0.7371 | 0.4222 |
| XGBoost | 0.6815 | 0.1771 | 0.3969 | 0.1378 | 0.7486 | 0.4018 |
| **Ensemble Avg** | **0.6822** | **0.1798** | **0.3992** | **0.1365** | **0.7698** | **0.2998** |

### 🦴 Stroke

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6561 | 0.0543 | 0.1956 | 0.0530 | 0.5965 | 0.0433 |
| CatBoost | 0.6796 | 0.0626 | 0.1960 | 0.0551 | 0.5424 | 0.4659 |
| XGBoost | 0.6827 | 0.0631 | 0.1971 | 0.0562 | 0.5278 | 0.4776 |
| **Ensemble Avg** | **0.6835** | **0.0641** | **0.1999** | **0.0560** | **0.5585** | **0.3231** |

### 🫀 High Blood Pressure (HIBP)

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6000 | 0.2504 | 0.5625 | 0.2162 | 0.9383 | 0.2094 |
| CatBoost | 0.6170 | 0.2678 | 0.5637 | 0.2113 | 0.9672 | 0.2969 |
| XGBoost | 0.6189 | 0.2721 | 0.5634 | 0.2190 | 0.9285 | 0.3464 |
| **Ensemble Avg** | **0.6194** | **0.2717** | **0.5654** | **0.2143** | **0.9579** | **0.2823** |

### 🦴 Arthritis

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.6397 | 0.3007 | 0.5603 | 0.2119 | 0.9516 | 0.2153 |
| CatBoost | 0.6473 | 0.3080 | 0.5648 | 0.2288 | 0.8923 | 0.3756 |
| XGBoost | 0.6424 | 0.3099 | 0.5619 | 0.2072 | 0.9822 | 0.2328 |
| **Ensemble Avg** | **0.6480** | **0.3130** | **0.5625** | **0.2077** | **0.9817** | **0.2503** |

### 🎗️ Cancer

| Model | ROC-AUC | PR-AUC | F2 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|
| LightGBM | 0.5662 | 0.0638 | 0.2182 | 0.0578 | 0.7128 | 0.0579 |
| CatBoost | 0.6014 | 0.0734 | 0.2382 | 0.0662 | 0.6794 | 0.4601 |
| XGBoost | 0.5936 | 0.0711 | 0.2340 | 0.0629 | 0.7314 | 0.4222 |
| **Ensemble Avg** | **0.5982** | **0.0723** | **0.2355** | **0.0641** | **0.7119** | **0.3115** |

---

## Ensemble ROC-AUC Ranking

| Rank | Disease | Ensemble ROC-AUC |
|---|---|---|
| 1 | Memory | 0.8332 |
| 2 | Lung | 0.7345 |
| 3 | Psychiatric | 0.7309 |
| 4 | Diabetes | 0.7030 |
| 5 | Stroke | 0.6835 |
| 6 | CVD | 0.6822 |
| 7 | Arthritis | 0.6480 |
| 8 | HIBP | 0.6194 |
| 9 | Cancer | 0.5982 |

---

## SHAP Feature Importance Plots

### Diabetes
![SHAP summary plot for Diabetes](shap_diabetes.png)

### Cardiovascular Disease (CVD)
![SHAP summary plot for CVD](shap_cvd.png)

### Stroke
![SHAP summary plot for Stroke](shap_stroke.png)

### Lung Disease
![SHAP summary plot for Lung Disease](shap_lung.png)

### Cancer
![SHAP summary plot for Cancer](shap_cancer.png)

### High Blood Pressure (HIBP)
![SHAP summary plot for HIBP](shap_hibp.png)

### Arthritis
![SHAP summary plot for Arthritis](shap_arthritis.png)

### Psychiatric Conditions
![SHAP summary plot for Psychiatric Conditions](shap_psychiatric.png)

### Memory Disease
![SHAP summary plot for Memory Disease](shap_memory.png)
