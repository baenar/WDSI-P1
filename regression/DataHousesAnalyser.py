from regression.DataHouses import DataHouses

if __name__ == '__main__':
    data = DataHouses()
    data.print_data_summary(output_file_path="regression/analysis/analysis_before.csv")

    data.clean_houses_data()
    data.print_data_summary(output_file_path="regression/analysis/analysis_after_wo_encoding.csv")

    data.clean_houses_data(do_encode_categorical_features=True)
    data.print_data_summary(output_file_path="regression/analysis/analysis_after_w_encoding.csv")
