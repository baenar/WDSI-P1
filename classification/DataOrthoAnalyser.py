import sys
import os

from classification.DataOrtho import DataOrtho

if __name__ == '__main__':
    data = DataOrtho()
    data.print_data_summary(output_file_path="classification/analysis/analysis_before.csv")

    data.clean_ortho_data(encode_target=True)
    data.print_data_summary(output_file_path="classification/analysis/analysis_after_encoding.csv")
    
    X_train, X_test, y_train, y_test = data.get_default_learn_test_data_split(target_column="growth direction")
    X_train.to_csv("classification/analysis/default_X_train.csv", index=False)
    X_test.to_csv("classification/analysis/default_X_test.csv", index=False)
    y_train.to_csv("classification/analysis/default_y_train.csv", index=False)
    y_test.to_csv("classification/analysis/default_y_test.csv", index=False)
    print("Successfully saved X_train, X_test, y_train, y_test to 'classification/analysis/'.")