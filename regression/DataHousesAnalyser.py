import os

from regression.DataHouses import DataHouses
from regression.DataHousesConfig import RESULTS_DIR, MODELS_CONFIG
from regression.DataHousesEvaluation import (
    run_hyperparameter_search,
    evaluate_best_model_cv,
    evaluate_on_test,
)
from regression.DataHousesReport import save_results_to_txt
from regression.DataHousesPlots import (
    plot_cv_metric_comparison,
    plot_cv_boxplots,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_test_metrics_heatmap,
)

if __name__ == '__main__':
    data = DataHouses()
    # data.print_data_summary(output_file_path=f"{ANALYSIS_DIR}/analysis_before.csv")

    data.clean_houses_data()
    # data.print_data_summary(output_file_path=f"{ANALYSIS_DIR}/analysis_after_wo_encoding.csv")

    data.clean_houses_data(do_encode_categorical_features=True)
    # data.print_data_summary(output_file_path=f"{ANALYSIS_DIR}/analysis_after_w_encoding.csv")

    X_train, X_test, y_train, y_test = data.get_default_learn_test_data_split("SalePrice")

    # X_train.to_csv(f"{ANALYSIS_DIR}/default_X_train.csv", index=False)
    # X_test.to_csv(f"{ANALYSIS_DIR}/default_X_test.csv", index=False)
    # y_train.to_csv(f"{ANALYSIS_DIR}/default_y_train.csv", index=False)
    # y_test.to_csv(f"{ANALYSIS_DIR}/default_y_test.csv", index=False)
    # print("Successfully saved X_train, X_test, y_train, y_test to 'regression/analysis/'.")

    # Tuning and evaluation
    all_results = {}

    for name, config in MODELS_CONFIG.items():
        search = run_hyperparameter_search(name, config, X_train, y_train)
        cv_results = evaluate_best_model_cv(name, search.best_estimator_, X_train, y_train)
        test_metrics, y_test_pred = evaluate_on_test(search.best_estimator_, X_test, y_test)

        all_results[name] = {
            "search_type": config["search_type"],
            "best_params": search.best_params_,
            "cv_results": cv_results,
            "test_metrics": test_metrics,
            "y_test_pred": y_test_pred,
        }

    # Saving results to textfiles
    save_results_to_txt(all_results, f"{RESULTS_DIR}/regression_results.txt")

    # Plot generation
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_cv_metric_comparison(all_results, RESULTS_DIR)
    plot_cv_boxplots(all_results, RESULTS_DIR)
    plot_predicted_vs_actual(all_results, y_test, RESULTS_DIR)
    plot_residuals(all_results, y_test, RESULTS_DIR)
    plot_test_metrics_heatmap(all_results, RESULTS_DIR)
    data.print_data_summary(output_file_path=f"{RESULTS_DIR}/analysis_after_w_encoding.csv")
    X_train.to_csv(f"{RESULTS_DIR}/default_X_train.csv", index=False)
    X_test.to_csv(f"{RESULTS_DIR}/default_X_test.csv", index=False)
    y_train.to_csv(f"{RESULTS_DIR}/default_y_train.csv", index=False)
    y_test.to_csv(f"{RESULTS_DIR}/default_y_test.csv", index=False)

