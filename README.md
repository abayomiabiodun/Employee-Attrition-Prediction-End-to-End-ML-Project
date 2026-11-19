# Employee-Attrition-Prediction-End-to-End-ML-Project

A leakage-proof, fully reproducible machine-learning project that identifies **which employees are at high risk of leaving** and **why**, so HR can act *before* the resignation happens. It is built on the IBM HR Analytics Employee Attrition & Performance dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange)
![Notebook](https://img.shields.io/badge/Jupyter-End--to--End-brightgreen)

---

## Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Key Results](#-key-results)
- [Why People Leave: Key Findings](#-why-people-leave-key-findings)
- [Recommendations](#-recommendations)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)
- [Limitations](#-limitations)
- [Author](#-author)

---

## Problem Statement

Employee attrition is expensive: recruiting, onboarding, lost productivity and lost knowledge can add up to a large fraction of a leaver's annual salary. HR usually learns about a resignation **after** it happens, when it is too late to act.

> *Can we identify, in advance, which current employees are at high risk of leaving, and understand why?*

| Item | Definition |
|---|---|
| **Unit of analysis** | One employee (1,470 rows) |
| **Target** | `Attrition` (`Yes` = 1, `No` = 0) |
| **Problem type** | Supervised binary classification, **imbalanced** (≈ 16% positives) |
| **Output** | A calibrated probability of leaving, a risk tier, and the main drivers |

**Success criteria**
- **Ranking quality:** ROC-AUC and PR-AUC. Accuracy is *not* used, because predicting "nobody leaves" already gives ~84%.
- **Operational quality:** recall (leavers caught) vs. precision (wasted interventions) at a chosen threshold
- **Generalisation:** small train–CV gap, test consistent with CV, and a score clearly above a label-shuffled baseline
- **Trustworthiness:** zero data leakage, calibrated probabilities, interpretable drivers

---

## Dataset

- **File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv` (IBM HR Analytics Employee Attrition & Performance)
- **Size:** 1,470 employees · 30 usable features after cleaning · 16.1% attrition
- **Features:** demographics, job/role, compensation, tenure, satisfaction surveys, work pattern (overtime, travel)

> The IBM dataset is **fictional and synthetic**, which makes it ideal for learning but cleaner than real HR data. It is also a single snapshot with no timestamps.

---

## Approach

```
raw CSV → audit & clean → stratified train/test split (test set locked away)
        → everything below is FIT ON TRAIN FOLDS ONLY
          feature engineering → preprocessing & reduction → imbalance handling → model
        → nested CV tuning → fit diagnostics → leakage audit
        → calibration & threshold choice → one-time test evaluation → interpretation
```

**Notebook sections**
1. Problem statement ·
2. Proposed solution ·
3. Setup ·
4. Data audit & cleaning ·
5. Train/test split ·
6. EDA with FDR-corrected feature screening ·
7. Feature engineering ·
8. Preprocessing & feature reduction ·
9. Model benchmarking ·
10. Tuning & nested CV ·
11. Under/over-fit diagnostics ·
12. Leakage audit ·
13. Calibration & thresholds ·
14. Final test evaluation ·
15. Interpretation ·
16. Business impact ·
17. Recommendations

**Engineered features** (each defined mathematically in the notebook)
- Career-shape ratios (tenure relative to career length, stagnation)
- **Peer-relative pay:** compa-ratio and a shrunk z-score within role × level (fitted on training data only)
- Satisfaction profile across the five 1 to 4 survey items
- Interaction and flag features motivated by EDA (e.g. single × overtime, overtime × travel, early-tenure flag)
- Yeo–Johnson power transform for skewed income and tenure variables

**Feature reduction (inside CV):** correlation filter, VIF pruning, and selection methods, with ablations showing whether engineering actually helps.

**Benchmarking:** 8 algorithms × 3 feature sets, and three imbalance strategies (none vs. class weights vs. SMOTE).

**Final model:** tuned, class-weighted, sigmoid-calibrated **L2 logistic regression** on the full feature set.

---

## Key Results

| Metric | Value |
|---|---|
| Cross-validated ROC-AUC (fresh folds) | 0.842 |
| Nested-CV ROC-AUC | 0.833 ± 0.029 |
| **Held-out test ROC-AUC** | **0.806** (95% CI 0.731 to 0.871) |
| **Held-out test PR-AUC** | **0.526** (chance = 0.160) |
| Train–CV gap | +0.064 AUC → good fit |
| Label-permutation test | p = 0.0244 |
| Top-decile capture | **36%** of leavers in the top 10% of risk scores; **66%** in the top 30% |

**Operating points on the test set**

| Threshold strategy | Threshold | Precision | Recall |
|---|---|---|---|
| Max-F1 | 0.37 | 0.55 | 0.36 |
| Recall-first | 0.15 | 0.35 | 0.72 |

---

## Why People Leave: Key Findings

1. **Overtime is the clearest single lever.** Attrition is roughly 2.5 to 3× higher among people working overtime. In the interpretation model, overtime multiplies the odds of leaving by about 4, holding other features fixed.
2. **Risk is concentrated early in a role and early in tenure.** `YearsInCurrentRole` is the top feature by permutation importance, followed by `YearsAtCompany` and `TotalWorkingYears`. The high-risk group has a median tenure of ~3 years vs ~6 for everyone else.
3. **Lower pay, junior level and no equity.** The high-risk group has a median income of ~3.5k vs 5.0k, and ~77% have no stock options vs 36%.
4. **Role and profile matter.** Sales representatives, laboratory technicians and HR roles run above average, and single employees and frequent travellers are over-represented among high-risk employees.
5. **Satisfaction surveys add real but secondary signal.** They overlap with tenure and pay information, so treat them as supporting evidence rather than primary drivers.

---

## Recommendations

| Priority | Action | Why |
|---|---|---|
| 1 | **Review overtime load and travel**; cap or compensate persistent overtime | Strongest, most actionable driver |
| 2 | **Strengthen onboarding and the first 2 to 3 years** (check-ins, mentoring, career paths, role rotation) | Risk peaks early in tenure and in the current role |
| 3 | **Fix junior-level pay competitiveness** using the compa-ratio; extend equity eligibility | Leavers are paid less and rarely hold options |
| 4 | **Targeted retention for sales reps and lab technicians** | Role-level attrition is far above average |
| 5 | **Use the model as a prioritisation aid, not a verdict**; pair every flag with a manager conversation | Precision and recall are moderate |
| 6 | **Monitor and recalibrate quarterly** | Drift is expected in any HR system |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install numpy pandas scipy scikit-learn statsmodels xgboost lightgbm imbalanced-learn shap matplotlib seaborn joblib jupyter
```

### 3. Add the data
Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` next to the notebook.

### 4. Run the notebook
```bash
jupyter notebook Employee_Attrition_Prediction.ipynb
```
Choose **Kernel → Restart & Run All**.

### Outputs
The notebook writes to `artifacts/`:
- `attrition_model.joblib`: the calibrated pipeline, both thresholds and metadata (loading it requires the notebook's custom classes `HRFeatureEngineer`, `CorrelationFilter` and `VIFFilter` to be defined)
- `test_set_risk_scores.csv`: test-set employees ranked by `P_leave` with risk tiers

---

## Repository Structure

```
├── Employee_Attrition_Prediction.ipynb        # Full end-to-end notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv      # IBM HR dataset (add locally)
├── artifacts/                                 # Generated by the notebook
│   ├── attrition_model.joblib
│   └── test_set_risk_scores.csv
└── README.md
```

---

## Limitations

- **Association, not causation.** The drivers predict who leaves but don't prove that changing them will keep people. Pilot interventions and measure the effect.
- **Test AUC (0.81) is lower than CV (0.84).** With few test leavers the interval is wide, so quote the range, not the point value.
- **Moderate recall at a precise threshold.** At max-F1 the model finds about a third of leavers; at the recall-first point it finds most but flags about a third of staff.
- **Synthetic snapshot data.** There are no dates, so no forward-in-time test is possible, and some features may have been recorded at the time of leaving. Expect lower performance on real HR data.
- **Cost analysis uses placeholder assumptions.** Replace them with your own numbers.
- **The saved model is refit on all 1,470 rows.** The reported test metrics refer to the train-only model.

**Next steps:** collect time-stamped data (snapshots ~6 months before the outcome), add exit-interview and engagement-survey history, and run a controlled retention pilot on the high-risk tier.

---

## Author

**Abayomi Abiodun**
Senior Data Scientist · Founder, Data Fanatics Community

If you found this useful, feel free to star the repo and open an issue with questions or suggestions.
