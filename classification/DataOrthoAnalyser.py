import sys
import os
import io

from sklearn import datasets
from classification.DataOrtho import DataOrtho
from classification.OrthoModel import OrthoModel

def analyze_ortho_data(data: DataOrtho, output_dir: str) -> None:
    df = data.get_dataframe()
    with open(output_dir, 'w') as f:
        f.write("--- DATAFRAME INFO ---\n")
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\n--- SUMMARY STATISTICS ---\n")
        f.write(df.describe().T.to_string())
    print("Report saved to data_report.txt")

if __name__ == '__main__':
    data = DataOrtho()

    data.encode_target_variable()
    data.generate_kde_plots()

    analyze_ortho_data(data, 'classification/analysis/data_report.txt')
    data.generate_correlation_matrix(filename='correlation_heatmap.png')

    #data.engineer_growth_features(drop_age_12=True)
    #data.generate_kde_plots(prefixes_to_plot=['delta_'])
    data.remove_highly_correlated_features(threshold=0.9)

    analyze_ortho_data(data, 'classification/analysis/after_engineering_report.txt')
    data.generate_correlation_matrix(filename='after_engineering_correlation_heatmap.png')

    X = data.df.drop(columns=["growth direction"])
    y = data.df["growth direction"]

    random_state = 123
    model = OrthoModel(random_state=random_state)
    model.train_and_evaluate_models(X=X, y=y, random_state=random_state)