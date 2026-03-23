import sys
import os
import io

from sklearn import datasets
from classification.DataOrtho import DataOrtho

if __name__ == '__main__':
    data = DataOrtho()

    data.clean_ortho_data(encode_target=True)

    df = data.get_dataframe()
    with open('classification/analysis/data_report.txt', 'w') as f:
        f.write("--- DATAFRAME INFO ---\n")
        buffer = io.StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        f.write("\n\n--- SUMMARY STATISTICS ---\n")
        f.write(df.describe().T.to_string())
    print("Report saved to data_report.txt")

    data.generate_visualizations()
    
    X_train, X_test, y_train, y_test = data.get_default_learn_test_data_split(target_column="growth direction")
    X_train.to_csv("classification/data/default_X_train.csv", index=False)
    X_test.to_csv("classification/data/default_X_test.csv", index=False)
    y_train.to_csv("classification/data/default_y_train.csv", index=False)
    y_test.to_csv("classification/data/default_y_test.csv", index=False)
    print("Successfully saved X_train, X_test, y_train, y_test to 'classification/data/'.")