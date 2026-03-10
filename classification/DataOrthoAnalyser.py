import sys
import os

from classification.DataOrtho import DataOrtho

if __name__ == '__main__':
    data = DataOrtho()
    data.print_data_summary(output_file_path="classification/analysis/analysis_before.csv")

    data.clean_ortho_data(encode_target=True)
    data.print_data_summary(output_file_path="classification/analysis/analysis_after_encoding.csv")
    
    X, y = data.get_features_and_target()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")