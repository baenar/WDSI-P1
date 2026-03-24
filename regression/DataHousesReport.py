import os
import numpy as np


def save_results_to_txt(all_results, output_path):
    """Saves full analysis results to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  REGRESSION MODEL ANALYSIS — HYPERPARAMETER TUNING & EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Cross-validation: RepeatedKFold (k=5, repeats=3, total=15 folds)\n")
        f.write(f"  Metrics: R², MAE, RMSE, MAPE, MedAE\n")
        f.write("=" * 80 + "\n\n")

        for name, res in all_results.items():
            f.write(f"{'─'*80}\n")
            f.write(f"  MODEL: {name}\n")
            f.write(f"{'─'*80}\n\n")

            f.write(f"  Search type:  {res['search_type'].upper()}\n")
            f.write(f"  Best params:  {res['best_params']}\n\n")

            f.write(f"  CROSS-VALIDATION RESULTS (RepeatedKFold, 15 folds):\n")
            f.write(f"  {'Metric':<10} {'Train (mean±std)':>25} {'Val (mean±std)':>25}\n")
            f.write(f"  {'-'*60}\n")

            cv = res["cv_results"]
            for metric in ["R2", "MAE", "RMSE", "MAPE", "MedAE"]:
                train_key = f"train_{metric}"
                test_key = f"test_{metric}"

                train_vals = cv[train_key]
                val_vals = cv[test_key]

                if metric != "R2":
                    train_vals = -train_vals
                    val_vals = -val_vals

                if metric == "RMSE":
                    train_vals = np.sqrt(train_vals)
                    val_vals = np.sqrt(val_vals)

                f.write(f"  {metric:<10} {train_vals.mean():>12.2f} ± {train_vals.std():>8.2f}"
                        f" {val_vals.mean():>12.2f} ± {val_vals.std():>8.2f}\n")

            f.write(f"\n  HOLD-OUT TEST SET RESULTS:\n")
            for metric, val in res["test_metrics"].items():
                f.write(f"    {metric:<10} {val:>12.4f}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("  SUMMARY — TEST SET COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"  {'Model':<25} {'R²':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>8} {'MedAE':>12}\n")
        f.write(f"  {'-'*77}\n")

        best_r2_name = None
        best_r2_val = -np.inf

        for name, res in all_results.items():
            m = res["test_metrics"]
            f.write(f"  {name:<25} {m['R2']:>8.4f} {m['MAE']:>12.2f} {m['RMSE']:>12.2f}"
                    f" {m['MAPE']:>8.4f} {m['MedAE']:>12.2f}\n")
            if m["R2"] > best_r2_val:
                best_r2_val = m["R2"]
                best_r2_name = name

        f.write(f"\n  BEST MODEL (by Test R²): {best_r2_name}\n")
        f.write(f"    R²:   {all_results[best_r2_name]['test_metrics']['R2']:.4f}\n")
        f.write(f"    RMSE: {all_results[best_r2_name]['test_metrics']['RMSE']:.2f}\n")
        f.write("=" * 80 + "\n")

    print(f"Results saved to: {output_path}")
