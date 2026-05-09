# Research methodology — FIFA 18 player classification (Football Analysis ML)

This document is intended as the **methods / metrics appendix** for a course paper or short journal-style report: every preprocessing decision, labeling rule, model family, tuning protocol, metric definition, and evaluation procedure used in `football_analysis` is enumerated here.

**Primary code references:** [`football_analysis/data.py`](./football_analysis/data.py), [`football_analysis/modeling.py`](./football_analysis/modeling.py), [`football_analysis/legacy.py`](./football_analysis/legacy.py), [`football_analysis/reporting.py`](./football_analysis/reporting.py).  
**Measured numbers** after each training/evaluation run appear in **`metrics.json`** and **`evaluation.json`** under your chosen output directory (`artifacts/` by default).

---

## 1. Research question & tasks

**Input.** One row per professional player in FIFA 18, with ordinal/numeric ratings and a few categorical fields (preferred foot, work rates, etc.). Source file resolved at runtime: **`FootballAnalysisML/players_18.csv`** (see `DEFAULT_DATA_PATH`).

**Predictive targets (two tasks).**

| Task ID | Prediction | Interpretation |
|--------|-------------|----------------|
| `role` | 4 discrete classes | Tactical group: attack, midfield, defense, goalkeeper |
| `position` | *K* discrete classes (~12 here) | Merged outfield / lineup code inferred from FIFA `player_positions` |

Both are **multiclass supervised classification** problems evaluated on **one stratified held-out partition** (see §6).

---

## 2. Label construction (deterministic logic)

### 2.1 Primary FIFA position token

CSV column **`player_positions`** may list several roles separated by commas. The pipeline uses **only the first token**:

\[
\text{primary\_position} = \mathrm{trim}\bigl(\mathrm{split}(\texttt{player\_positions},\, \text{\texttt{`,`}})_0\bigr).
\]

If empty or NaN ⇒ row **cannot produce a supervised label** → dropped downstream.

**Implementation:** `extract_primary_position`, `add_targets`.

### 2.2 Role labels (`role_target`)

FIFA abbreviated codes (`ST`, `CDM`, `GK`, …) are mapped via **explicit finite sets**:

- **Attack (`attack`):** `{LS, ST, RS, LW, LF, CF, RF, RW}`
- **Midfield (`midfield`):** `{LAM, CAM, RAM, LM, LCM, CM, RCM, RM, LWB, LDM, CDM, RDM}`
- **Defense (`defense`):** `{RWB, LB, LCB, CB, RCB, RB}`
- **Goalkeeper (`goalkeeper`):** `{GK}`

Any primary code outside these sets ⇒ **that row is discarded** (`dropna` on `role_target`). This removes exotic or inconsistent codes rather than assigning an “unknown” class.

**Implementation:** `ROLE_ATTACK`, …, `ROLE_GOALKEEPER`, `map_role_target`.

### 2.3 Position labels (`position_target`)

Start from `primary_position`, then apply **deterministic merges** to stabilize rare codes:

| Raw code | Merged label |
|----------|----------------|
| `CF` | `ST` |
| `RWB` | `RB` |
| `LWB` | `LB` |

All others keep their abbreviation (e.g. `CAM`, `GK`, `CM`).

**Implementation:** `POSITION_MERGES`, `map_position_target`.

### 2.4 Class inventory on full data

Exact support per class varies (defensive/box players more frequent than niche wing mids). Inspect `classification_report.support` inside `metrics.json` after training for empirical counts **on your held-out set**.

---

## 3. Feature engineering

### 3.1 Fixed feature subsets

Two lists are authoritative:

**Numeric ratings & physical indicators** (`NUMERIC_FEATURES`):  
`age`, `overall`, `height_cm`, `weight_kg`, `potential`, `release_clause_eur`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, all listed attacking/movement/power/mental/defensive/goalkeeping sub-ratings (~47 numeric columns).

**Categorical descriptors** (`CATEGORICAL_FEATURES`):  
`preferred_foot`, `work_rate`, `weak_foot`, `skill_moves`, `international_reputation`.

Missing raw columns before pipeline fit would fault at load—the dataset is expected complete for these headers.

### 3.2 Missingness in numeric fields

Many football attributes are genuinely missing (e.g. release clause unknown). Procedure:

1. For each baseline numeric column \(c\) in `NUMERIC_FEATURES`, if \(\exists\) NA in \(c\) on the **model frame**, add **`c_missing` ∈ {0,1}** (1 ⇒ value was NA before imputation).
2. **`SimpleImputer(strategy="median")`** on numeric block (applied **after** indicator construction so the indicator carries information).
3. **`SimpleImputer(strategy="most_frequent")`** + **`OneHotEncoder(handle_unknown="ignore")`** on categoricals inside a **`ColumnTransformer`**.

**Motivation.** Indicators let tree models separate “measurement genuinely absent” vs “low measured skill” without falsely imputing medians alone.

### 3.3 Preprocessor isolation across search candidates

During hyperparameter search, **each estimator family receives a freshly built `ColumnTransformer`** so parallel CV workers never share mutated transformer state (**implementation detail for reproducibility**).

---

## 4. Modeling & hyperparameter tuning

### 4.1 Base learners

Two bagged randomized tree ensembles (scikit-learn):

| Family | Implementation | Inner parallelism |
|--------|----------------|-------------------|
| Random forest | `RandomForestClassifier(random_state=R, n_jobs=1)` | Deliberately 1 worker per tree estimator |
| Extremely randomized trees | `ExtraTreesClassifier` idem | 1 |

**Why `n_jobs=1` inside trees.** Parallelism is delegated to **`RandomizedSearchCV(n_jobs=…)`** so **different CV folds run on different CPUs** without oversaturating nested parallelism.

### 4.2 Pipelines & search

Combined pipeline:

`Pipeline([ ("preprocessor", ColumnTransformer(...) ), ("model", forest) ])`

**Optimizer:** **`RandomizedSearchCV`**

| Setting | Meaning |
|---------|---------|
| `param_distributions` | Per-model grids (`model__…` prefixed) |
| `n_iter` | Random trials **per estimator family** (CLI / preset) |
| `cv` | `StratifiedKFold(n_splits, shuffle=True, random_state=R)` |
| `scoring` | **`f1_macro`** (**macro-averaged F1** selects best configuration) |
| `refit` | `True`: refit best params on **all train rows** after search |
| `random_state` | Fixed across runs for stochastic sampling + tree randomness |

**Two search presets.**

| Preset ID | Intended use | Role grid extras | Position grid behavior |
|-----------|---------------|------------------|-------------------------|
| `fast` | Development / notebooks | `[200 … 500]` trees | Reduced depth & narrower grid for speed |
| `paper` | Maximum reported search fidelity | Adds **`600`** trees candidate + full depth options | Uses **same full grids as role**, including **`max_depth=None`** (expensive fits) |

**CLI defaults (when omitting overrides).**

| Task | `--preset fast` iterations × folds | `--preset paper` iterations × folds |
|------|-------------------------------------|--------------------------------------|
| `train_role` | 8 × 5 | **28 × 5** |
| `train_position` | 5 × 3 | **28 × 5** |

Raise `--search-iterations` further for diminishing returns (more wall-clock).

### 4.3 Reported grids (conceptual)

Both families share the qualitative axes below; exact lists live in **`model_candidates()`** in [`modeling.py`](./football_analysis/modeling.py):

- **`n_estimators`:** bag size (trees / extremely randomized splits)
- **`max_depth`:** tree depth ceiling; **`None` ⇒ grow until purity / min_leaf** (much slower especially for `position`)
- **`min_samples_split` / `min_samples_leaf`**
- **`max_features`:** fractional or `sqrt` / `log2` feature subsampling at each split
- **`class_weight`:** imbalance handling (`None`, `balanced`, `balanced_subsample` for RF; ET omits subsample variant)

Winner = family with **best cross-validated `f1_macro`**. Tie-breaking is implicit sklearn ordering if scores equal—rare given floating variance.

---

## 5. Train / test split protocol

Executed once per `train_best_model(...)` invocation:

```
(X, y_encoded, row_ids) = stratified_shuffle_split(
      test_fraction = test_size (=0.2 default),
      stratify=y_encoded,
      random_state=R (=15 CLI default)
)
```

- **`row_ids`** = integer positions into the dataframe returned by **`build_model_frame`** (post–target drop).
- Persisted artifact fields: **`test_indices`**, **`label_classes`**, fitted **`estimator`**, feature lists, **`model_name`**, **`search_preset`**.

**Leakage safeguards.** Targets never enter unsupervised preprocessing prior to stratification—the frame already contains supervised columns but split happens **before** CV on **train subset only**.

---

## 6. Evaluation & exported metrics

### 6.1 Primary held-out scoring (during training)

Immediately after fitting the globally best pipeline, sklearn predictions on **`X_test`**, **`y_test`** encoded with the **same LabelEncoder**:

| Key (JSON / dict) | Definition | Typical use in paper |
|-------------------|-------------|-----------------------|
| `accuracy` | \(\frac{1}{n}\sum \mathbb{1}[\hat y_i = y_i]\) | Easy headline; inflated when majority class dominates |
| `balanced_accuracy` | Class-wise recall averaged uniformly | Robust to imbalance summaries |
| `macro_f1` | Unweighted mean of **per-class F1** | Harmonizes recall/precision uneven support |
| `weighted_f1` | F1 averaged weighted by **support on test set** | Aligns metric mass with empirical frequency |
| `cohen_kappa` | Agreement vs chance for categorical labels | Normalized consensus beyond prevalence |
| `log_loss` | Average \(-\log p_\theta(y_i \mid x_i)\) via `predict_proba` | Calibration-sensitive penalization of probability mass |

**Classification report.** Nested dict with `precision`, `recall`, `f1-score`, `support` per class (+ macro/weighted aggregates duplicate some scalars).

**Confusion matrix.** Square counts **rows=true encoded index**, **columns=pred encoded index** aligned with `confusion_matrix_labels` (= human-readable FIFA strings in lexical label order—not necessarily alphabetical string order).

### 6.2 Position-only **top-k accuracy**

Probability matrix columns follow **`LabelEncoder.classes_`** order (must match estimator `predict_proba` ordering).

Top-k correctness **k ∈ {3,5,10}** only if **`k ≤ n_classes`**:

\[
\text{Top-}k = \frac{1}{n_\text{test}} \sum_i \mathbf{1}\bigl[ y_i \in \mathrm{ArgsortDescending}(\hat{p}_i)[:k]\bigr].
\]

Useful narrative: **strict position code** is ambiguous in FIFA taxonomy; recruiters might accept plausible neighbors.

### 6.3 CLI re-evaluation (`evaluate_saved_model`)

`python -m football_analysis.evaluate --task {role|position}`:

1. Reload **raw CSV** (may differ path via `--data-path`; default constant).
2. Rebuild deterministic frame + targets (**must match preprocessing logic version** saved with model).
3. Slice **exact indexed rows**: `artifact["test_indices"]`.
4. Encode `y_true` via label order stored in **`label_classes`**.
5. Calls same `compute_metrics` → writes **`evaluation.json`**.

Embedded **`evaluation_protocol`** block clarifies parity requirements:

- Indices only valid relative to **`build_model_frame(load_raw(...))`** output ordering.
- **Changing CSV row order/content** breaks strict equality with original training diagnostics.

### 6.4 Legacy exploratory baseline (**role only** — not primary claim)

[`legacy.py`](./football_analysis/legacy.py):

- Loads `players_18_processed_classification.csv`.
- Drops target column **`Position 1`**, naive **`fillna(0)`**.
- Fits a **standalone** `RandomForestClassifier(n_estimators=200)` on **alternative engineered features**.
- Separate random split ⇒ **numbers are incomparable apples-to-pear mechanically** vs main pipeline—they contextualize uplift from deterministic raw-attribute mapping vs older processed workbook.

Baseline metrics appear optionally under **`legacy_baseline` in role `metrics.json`**.

---

## 7. Reproducibility matrix

Record these in publication supplement:

| Field | Stored where |
|-------|---------------|
| `random_state`, `test_size`, `search_iterations`, `cv_folds` | `metrics.json`, artifact dict |
| `search_preset`, `best_params`, CV scores/tie families | `metrics.json`, `candidate_results` |
| `train_rows`, `test_rows` | both |
| `label_classes`, `numeric_features`, `categorical_features` snapshots | **`model.joblib`** |

**Environment.**

- sklearn / numpy minor version drift can shift tie-break nuances—record `pip freeze` excerpt.
- **Parallel CV nondeterminism:** despite fixed seeds many BLAS backends yield tiny floating differences negligible at float32 boundary—macro metrics stable at 4 d.p. practically.

---

## 8. Suggested wording for Discussion / Limitations 

1. **Label simplification.** First-token mapping discards multitasking players’ secondary registrations.
2. **Class imbalance.** Macro-F1 penalizes rarity; interpret alongside weighted-F1 / per-class recall.
3. **Tree opacity.** Interpretability relies on permutation importance extensions (future work)—not bundled now.
4. **Year / game engine bias.** FIFA 18 ≠ modern league tracking data.
5. **Search stochasticity.** Randomized—not exhaustive—search; widen `n_iter` for publication-grade robustness curves.

---

## 9. Command cheat sheet (`paper` quality)

Training (maximum default search widths & iterations — **may take tens of minutes to hours**, especially `position:` **paper** preset):

```bash
pip install -r requirements.txt

# Role — paper preset (writes artifacts/role or choose --output-dir)
python -m football_analysis.train_role --preset paper

# Position — paper preset (long run; parallel CV recommended)
python -m football_analysis.train_position --preset paper

# Quieter sklearn internals
FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0 python -m football_analysis.train_position --preset paper

# Re-evaluate (writes evaluation.json)
python -m football_analysis.evaluate --task role
python -m football_analysis.evaluate --task position
```

Isolated reproducible bundles for the paper annex:

```bash
python -m football_analysis.train_role --preset paper --output-dir artifacts/paper_eval/role
python -m football_analysis.train_position --preset paper --output-dir artifacts/paper_eval/position
python -m football_analysis.evaluate --task role --artifact-dir artifacts/paper_eval
python -m football_analysis.evaluate --task position --artifact-dir artifacts/paper_eval
```

**Tests:**  
`python -m unittest discover -s tests -v`

---

## 10. Filling empirical tables for the manuscript

1. Train with desired preset / iterations.
2. Open **`artifacts/<task>/metrics.json`**.
3. Copy scalars (`accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `cohen_kappa`, `log_loss`, optionally `top_k_*`).
4. Export confusion matrix Figure from `confusion_matrix` + `labels`.
5. Cite methodological sections §2–§6 for reviewer traceability.

*This appendix is descriptive of code as of repo version under development; cite commit hash alongside PDF.*

---

## 11. Example empirical snapshot (`paper`-quality run, reproducible subdirectory)

Artifacts path: **`artifacts/paper_eval/`** (separate from default `artifacts/role`, `artifacts/position`).

**Training commands used for this snapshot**

```bash
FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0 python3 -m football_analysis.train_role \
  --preset paper --search-iterations 20 --cv-folds 5 \
  --output-dir artifacts/paper_eval/role --cv-jobs -1

FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0 python3 -m football_analysis.train_position \
  --preset paper --search-iterations 14 --cv-folds 5 \
  --output-dir artifacts/paper_eval/position --cv-jobs -1
```

*Stricter reproducibility ceiling:* increase `--search-iterations` toward the **`paper` preset defaults** (role **28**, position **28**) on hardware with ample RAM—the position search with unrestricted depth stresses memory; **`--cv-jobs 1`** trades speed for stability if joblib emits worker warnings.*

**Held-out metrics (training-time `metrics.json`, *n*=3591 test rows each)**

| Task | Accuracy | Balanced acc | Cohen κ | Macro F1 | Weighted F1 | Log loss | Winner | Extra |
|------|----------|---------------|---------|----------|-------------|----------|--------|-------|
| Role | **0.905** | **0.910** | **0.865** | **0.916** | **0.905** | **0.252** | **ExtraTrees** | — |
| Position | **0.744** | **0.613** | **0.714** | **0.610** | **0.738** | **0.781** | **ExtraTrees** | top-3 **0.929**, top-5 **0.984**, top-10 **1.000** |

**CLI re-check (`evaluation.json`):** identities match **`metrics.json`** scalars row-for-row — confirms bitwise parity of deterministic evaluation path vs training-time holdout slicing.

**Legacy baseline (`metrics.json → legacy_baseline`, separate split & feature space):**

- Accuracy ≈ **0.654**, macro-F1 ≈ **0.701**.

(Full decimals, confusion matrix, classification report JSON, hyperparameters ⇒ open `artifacts/paper_eval/*/metrics.json` and `evaluation.json`.)

---

## 12. Neural-network extension (`--backend torch`)

The repository now includes an optional PyTorch tabular model for both tasks:

- command path: `python -m football_analysis.train_role --backend torch` and `python -m football_analysis.train_position --backend torch`
- implementation: `football_analysis/nn_tabular.py`
- architecture: numeric branch (median imputation + standardization on train split), categorical branch (learned embeddings), then MLP head (`Linear -> BatchNorm -> ReLU -> Dropout`)
- optimization: `AdamW`, early stopping by validation macro-F1, deterministic seeds (`numpy` + `torch`)
- class imbalance: weighted cross-entropy for `position`

### 12.1 Artifact format and parity

- torch training writes `model.pt` and `metrics.json` under task output directory
- `metrics.json` keeps the same core fields used by sklearn path (`accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `cohen_kappa`, `log_loss`, confusion matrix, and top-k for `position`)
- `python -m football_analysis.evaluate --task <task>` auto-detects `model.pt`; pass `--backend torch` or `--backend sklearn` to force backend

### 12.2 Suggested comparison protocol for manuscript tables

1. Train sklearn baseline (`--backend sklearn`, default).
2. Train neural model with the same random seed and `--sample-size` policy.
3. Compare `metrics.json` side-by-side for each task.
4. Report both strict metrics (accuracy, macro-F1) and position ambiguity metrics (top-3/top-5 accuracy).
