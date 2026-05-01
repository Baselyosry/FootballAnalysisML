# Football Analysis ML — Project evaluation snapshot

Short report for graders; **full methods, definitions, grids, evaluation protocol ⇒** **[`RESEARCH_METHODOLOGY_AND_METRICS.md`](./RESEARCH_METHODOLOGY_AND_METRICS.md)** (paper-ready appendix).

---

## Deliverables overview

| Path | Contents |
|------|-----------|
| `football_analysis/` | Data labels, preprocessing, randomized search training, persistence, evaluation helpers |
| `tests/` | Determinism checks, preprocessor numerics, end-to-end smoke + `preset=paper` grid path |
| `FootballAnalysisML/players_18.csv` | Raw FIFA 18–style spreadsheet (training source of truth) |
| **`RESEARCH_METHODOLOGY_AND_METRICS.md`** | **Detailed research paper appendix** — every labeling rule, feature, metric formula, preset, caveat |
| `PROJECT_EVALUATION_REPORT.md` (this file) | Executive numbers + quickest reproduce commands |

---

## Maximum-accuracy preset (`paper`)

CLI flags **`--preset paper`** widen tree hyperparameter grids and raise default search budget:

| Trainer | `--preset fast` (default CLI iters × folds) | `--preset paper` default iters × folds |
|---------|--------------------------------------------|----------------------------------------|
| `train_role` | 8 × 5 | **28 × 5** |
| `train_position` | 5 × 3 | **28 × 5** |

Overrides: `--search-iterations N`, `--cv-folds K` (omit to use preset table above).

Artifacts from a **`paper`** run retain `search_preset: "paper"` inside `metrics.json` / serialized model bundle.

Parallel CV: **`--cv-jobs -1`**. Quiet logs: **`FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0`**.

---

## Latest measured scores (recommended paper bundle)

Repro subdirectory **`artifacts/paper_eval/`** trained with **`paper`** grids (**role:** 20 search trials × 5 folds; **position:** 14 trials × 5 folds — strong but not exhausting default 28/28 ceilings). **Held-out rows:** 3591.

| Task | Accuracy | Balanced acc | Cohen κ | Macro F1 | Weighted F1 | Log loss | Best model |
|------|----------|---------------|---------|----------|-------------|----------|------------|
| Role | **0.905** | **0.910** | **0.865** | **0.916** | **0.905** | **0.252** | ExtraTrees |
| Position | **0.744** | **0.613** | **0.714** | **0.610** | **0.738** | **0.781** | ExtraTrees |

**Position ranking metrics (same split):** top-3 **0.929** · top-5 **0.984** · top-10 **1.000**.

**Exported metrics file keys** (`metrics.json`, mirrored on `evaluation.json` after `evaluate`): `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `cohen_kappa`, `log_loss`, `classification_report`, `confusion_matrix`, (+ `top_{3,5,10}_accuracy` for position), CV metadata, tuned parameters.

**Baseline note (alternative legacy CSV RF).** Stored as `legacy_baseline` alongside role **`paper`** artifacts: accuracy ≈ **0.654**, macro-F1 ≈ **0.701** (**not** comparable split/features to main pipeline—see appendix §6.4).

---

## Fastest reproducibility block

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v

# Recommended paper-grade bundle (isolate output paths)
mkdir -p artifacts/paper_eval/{role,position}
FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0 python -m football_analysis.train_role \
  --preset paper --output-dir artifacts/paper_eval/role
FOOTBALL_ANALYSIS_SEARCH_VERBOSE=0 python -m football_analysis.train_position \
  --preset paper --output-dir artifacts/paper_eval/position

python -m football_analysis.evaluate --task role --artifact-dir artifacts/paper_eval
python -m football_analysis.evaluate --task position --artifact-dir artifacts/paper_eval
```

*(Position `paper` default may run a long time; lower memory pressure with `--cv-jobs 1` if needed.)*

---

## Operational fixes (prior iteration)

Previously, **position fast defaults** suppressed progress and exploded wall-clock (`max_depth=None` + enormous forests). **`fast`/`paper` presets**, parallel CV folds, preprocessor isolation per candidate, flushed logging, and narrower **`fast`** position grid solved observability/time-to-result while **`paper`** retains **full** exploratory capacity for manuscripts.

Cross-reference **`README.md`** for notebook & layout.
