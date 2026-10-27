# Competition-organizer source code (attribution)

These files originated from [NYUSH-ML/ml-competition-sp26](https://github.com/NYUSH-ML/ml-competition-sp26), the official Spring 2026 CSI500 Stock Selection Competition repository. As compared on 2026-10-25, they are byte-for-byte identical to files in the organizer's `main` branch and are **not original project code**.

| Local file | Official upstream source | Purpose |
| --- | --- | --- |
| `baseline_xgboost.py` | [baseline_xgboost.py](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/baseline_xgboost.py) | Starting XGBoost baseline used for comparison. |
| `score_submission.py` | [score_submission.py](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/score_submission.py) | Evaluate returns relative to CSI500. |
| `validate_submission.py` | [validate_submission.py](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/validate_submission.py) | Check portfolio constraints before submission. |

**Missing upstream supporting files:** The baseline requires the organizer's [`features.py`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/features.py), which is not included in this repository. The upstream project also supplies [`download_data.py`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/download_data.py) and [`requirements.txt`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/requirements.txt). These can be retrieved from upstream when needed. Local input files belong under the repository root's `data/` directory.

**Path note:** The copy of `score_submission.py` is made compatible with the new folder layout by changing only its `DATA_DIR` path to point to the repository root. The baseline is also relocated with its `DATA_DIR` path adjusted. `validate_submission.py` remains byte-for-byte unmodified. Thus file identity to upstream refers to the pre-reorganization copies in the original repository; after the migration, the two path-adjusted files have minor layout-only changes. These changes do not alter scoring or model algorithms.

Usage from repository root:

```bash
python organizer/validate_submission.py results/final_self_ensemble.csv
python organizer/score_submission.py --help
```

See the [main README](../README.md) for the model pipeline and final results.
