from regression.DataHouses import DataHouses

if __name__ == '__main__':
    data = DataHouses()
    data.print_data_summary(output_file_path="regression/analysis/analysis_before.csv")

    data.clean_houses_data()
    data.print_data_summary(output_file_path="regression/analysis/analysis_after_wo_encoding.csv")

    data.clean_houses_data(do_encode_categorical_features=True)
    data.print_data_summary(output_file_path="regression/analysis/analysis_after_w_encoding.csv")

    X_train, X_test, y_train, y_test = data.get_default_learn_test_data_split("SalePrice")

    X_train.to_csv("regression/analysis/default_X_train.csv", index=False)
    X_test.to_csv("regression/analysis/default_X_test.csv", index=False)
    y_train.to_csv("regression/analysis/default_y_train.csv", index=False)
    y_test.to_csv("regression/analysis/default_y_test.csv", index=False)
    print("Successfully saved X_train, X_test, y_train, y_test to 'regression/analysis/'.")
