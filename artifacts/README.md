# Shared evaluation artifacts

**Tracked in Git (`.json` only)** — teammates can read headline scores without training:

| Path | Contents |
|------|-----------|
| `paper_eval/role/` · `paper_eval/position/` | **`paper`** preset run (`metrics.json`, `evaluation.json`) |
| `role/` · `position/` | **`fast`** preset default artifact summaries |

**Not tracked** — `.joblib` fitted pipelines exceed GitHub’s file-size limits (often hundreds of MB). Regenerate locally:

```bash
python -m football_analysis.train_role --preset paper --output-dir artifacts/paper_eval/role
python -m football_analysis.train_position --preset paper --output-dir artifacts/paper_eval/position
```

Smoke-test folders (`artifacts/test_*`) remain ignored.
