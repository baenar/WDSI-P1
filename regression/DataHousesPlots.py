import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')

def plot_cv_metric_comparison(all_results, output_dir):
    """Bar chart comparing CV validation metrics across models."""
    models = list(all_results.keys())
    metrics_to_plot = ["R2", "MAE", "RMSE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-Validation Metrics Comparison (RepeatedKFold)", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics_to_plot):
        means = []
        stds = []
        for name in models:
            cv = all_results[name]["cv_results"]
            vals = cv[f"test_{metric}"]
            if metric != "R2":
                vals = -vals
            if metric == "RMSE":
                vals = np.sqrt(vals)
            means.append(vals.mean())
            stds.append(vals.std())

        bars = ax.bar(models, means, yerr=stds, capsize=5, color=sns.color_palette("muted", len(models)),
                      edgecolor="black", linewidth=0.5)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{m:.2f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_cv_boxplots(all_results, output_dir):
    """Box plots showing distribution of CV fold scores for each model."""
    metrics_to_plot = ["R2", "RMSE", "MAE"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-Validation Score Distributions (15 folds)", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics_to_plot):
        data_for_plot = []
        for name, res in all_results.items():
            cv = res["cv_results"]
            vals = cv[f"test_{metric}"]
            if metric != "R2":
                vals = -vals
            if metric == "RMSE":
                vals = np.sqrt(vals)
            for v in vals:
                data_for_plot.append({"Model": name, metric: v})

        df_plot = pd.DataFrame(data_for_plot)
        sns.boxplot(data=df_plot, x="Model", y=metric, ax=ax, palette="muted")
        ax.set_title(f"{metric} Distribution", fontsize=12)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_boxplots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_predicted_vs_actual(all_results, y_test, output_dir):
    """Scatter plots: predicted vs actual for each model on the test set."""
    n = len(all_results)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle("Predicted vs Actual — Test Set", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        y_pred = res["y_test_pred"]
        ax.scatter(y_test, y_pred, alpha=0.5, s=15, edgecolors="k", linewidths=0.3)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")

        ax.set_xlabel("Actual SalePrice")
        ax.set_ylabel("Predicted SalePrice")
        ax.set_title(f"{name}\nR²={res['test_metrics']['R2']:.4f}, RMSE={res['test_metrics']['RMSE']:.0f}")
        ax.legend(loc="upper left")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "predicted_vs_actual.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_residuals(all_results, y_test, output_dir):
    """Residual plots for each model on the test set."""
    n = len(all_results)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle("Residuals — Test Set", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        y_pred = res["y_test_pred"]
        residuals = y_test.values - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=15, edgecolors="k", linewidths=0.3)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted SalePrice")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name}")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_test_metrics_heatmap(all_results, output_dir):
    """Heatmap of test set metrics across models."""
    models = list(all_results.keys())
    metrics = ["R2", "MAE", "RMSE", "MAPE", "MedAE"]

    data = []
    for name in models:
        row = [all_results[name]["test_metrics"][m] for m in metrics]
        data.append(row)

    df_heat = pd.DataFrame(data, index=models, columns=metrics)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
    ax.set_title("Test Set Metrics Heatmap", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "test_metrics_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
