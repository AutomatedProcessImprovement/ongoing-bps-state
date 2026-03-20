import pandas as pd, json
from .clustered_short_term_simulation import train_clustering_models, apply_models_to_test, _to_jsonable

root = "outputs_confidence/LOAN_STABLE/su0f4f"
train = pd.read_csv(f"{root}/train_samples.csv")
test = pd.read_csv(f"{root}/test_samples.csv")
target_col = "err_RTD_mean"   # or "err_cycle_mean"

models = train_clustering_models(train, target_col=target_col)
with open(f"{root}/cluster_models.json", "w", encoding="utf-8") as fh:
    json.dump(_to_jsonable(models), fh, indent=2)

test_eval = apply_models_to_test(models, test, target_col=target_col)
test_eval.to_csv(f"{root}/test_evaluation.csv", index=False)
