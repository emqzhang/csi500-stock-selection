# Intermediate development


| File or directory | Purpose |
| --- | --- |
| `model_v2.py` | Intermediate LightGBM implementation with robust training and five-seed ensembling, preceding `../model_v2_portfolio_aware_safe.py`. |

Always invoke the scripts from the **repository root**, e.g. `python intermediate/model_v2.py --as-of 20260430 --out results/v2_example.csv`.

The final submitted CSV belongs under `../results/`. See [the main README](../README.md) for the workflow, commands, evaluation results, and missing data dependencies.
