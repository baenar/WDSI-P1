import os

import pandas as pd
import numpy as np
import operator
from typing import List, Dict, Any, Callable, Optional, Tuple


class DataManager:
    __slots__ = ['df']

    def __init__(self, file_path: str):
        """
        Initializes the DataManager object and loads data from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing the data.
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"Successfully loaded data from: {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at path {file_path}")
            self.df = None

    def get_columns_names(self) -> List[str]:
        """
        Returns a list of all column names in the dataset.
        """
        if self.df is not None:
            return self.df.columns.tolist()
        return []

    def get_column_values(self, column_name: str) -> Dict[Any, int]:
        """
        Counts and returns the occurrences of individual values in a given column.

        Args:
            column_name (str): The name of the column to analyze.

        Returns:
            Dict: A dictionary where the key is the column value and the value is its frequency.
        """
        if self.df is not None and column_name in self.df.columns:
            return self.df[column_name].value_counts().to_dict()
        return {}

    def remove_column_from_set(self, column_name: str) -> None:
        """
        Removes the specified column from the dataset.

        Args:
            column_name (str): The name of the column to remove.
        """
        if self.df is not None and column_name in self.df.columns:
            self.df.drop(columns=[column_name], inplace=True)

    def remove_rows_from_set_by_column_value(self, column_name: str, value: Any, op: Callable = operator.eq) -> None:
        """
        Removes rows that meet a given logical condition in a specific column.

        Args:
            column_name (str): The name of the column to filter by.
            value (Any): The reference value for the condition.
            op (Callable): A function from the `operator` module defining the condition (default is `operator.eq`).
                           You can pass e.g., `operator.gt` (>), `operator.lt` (<), `operator.ne` (!=).
        """
        if self.df is None or column_name not in self.df.columns:
            return
        condition = op(self.df[column_name], value)
        self.df = self.df[~condition]

    def map_values_in_column_by_dict(self, column_name: str, value_map: Dict[Any, Any]) -> None:
        """
        Maps values in a specific column based on a provided dictionary.
        Values that are not present in the dictionary keys will remain unchanged.

        Args:
            column_name (str): The name of the column to modify.
            value_map (Dict[Any, Any]): A dictionary mapping old values to new values.
        """
        if self.df is not None and column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].map(value_map).fillna(self.df[column_name])

    def replace_value_in_column(self, column_name: str, old_value: Any, new_value: Any) -> None:
        """
        Replaces all occurrences of the old value with the new one in the selected column.

        Args:
            column_name (str): The name of the column to modify.
            old_value (Any): The value to be replaced.
            new_value (Any): The new value.
        """
        if self.df is not None and column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].replace(old_value, new_value)

    def get_categorical_columns_names(self) -> List[str]:
        """
        Finds and returns a list of all columns that contain text.

        Returns:
            List[str]: A list of categorical column names.
        """
        if self.df is not None:
            return self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        return []

    def encode_categorical_feature(self, column_name: str) -> None:
        """
        Performs One-Hot Encoding on a single specified categorical column.
        Uses drop_first=True to avoid the dummy variable trap in linear regression.

        Args:
            column_name (str): The name of the column to encode.
        """
        if self.df is None:
            return

        if column_name in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=[column_name], drop_first=True, dtype=int)
        else:
            print(f"  Warning: Column '{column_name}' not found in the dataset.")

    def get_default_learn_test_data_split(self, target_column: str, test_size: float = 0.3) \
            -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """
        Splits the data into training and testing sets using Systematic Sampling.
        Sorts the dataset by all columns prioritizing the target and picks evenly spaced
        rows for the test set to ensure a highly uniform distribution.

        Args:
            target_column (str): The name of the target variable to predict.
            test_size (float): Proportion of the dataset to include in the test split, by default 0.3.

        Returns:
            Tuple: A tuple containing (X_train, X_test, y_train, y_test).
        """
        if self.df is None or target_column not in self.df.columns:
            print(f"Error: Target column '{target_column}' not found or dataset is empty.")
            return None, None, None, None

        print(f"Performing systematic split ({100 - test_size * 100:.0f}% train, {test_size * 100:.0f}% test)...")

        # SORTING DATAFRAME
        sort_cols = [target_column] + [col for col in self.df.columns if col != target_column]
        sorted_df = self.df.sort_values(by=sort_cols).reset_index(drop=True)

        # PREPARING INDICES OF WANTED TEST ROWS
        total_rows = len(sorted_df)
        num_test_samples = int(total_rows * test_size)
        test_indices = np.linspace(0, total_rows - 1, num_test_samples, dtype=int)
        is_test = np.zeros(total_rows, dtype=bool)
        is_test[test_indices] = True

        # SPLITTING DATAFRAME
        test_df = sorted_df[is_test]
        train_df = sorted_df[~is_test]

        # SPLITTING FOR X AND Y
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        return X_train, X_test, y_train, y_test

    def print_data_summary(self, max_unique_values: int = 20, output_file_path: Optional[str] = None) -> None:
        """
        Prints a summary of the data distribution for each column (numerical count and percentage).
        Uses the `get_column_values` method internally. Optionally saves the summary to a CSV.

        Args:
            max_unique_values (int): Maximum number of unique values to display fully.
            output_file_path (Optional[str]): Path to save the summary CSV. If None, only prints to console.
        """
        if self.df is None or self.df.empty:
            print("Error: Dataset is empty or not loaded.")
            return

        print("\n--- DATA DISTRIBUTION SUMMARY ---")

        csv_data = []

        for column in self.get_columns_names():
            print(f"\nColumn: '{column}'")

            counts_dict = self.get_column_values(column)

            if not counts_dict:
                print("  Column is empty (only NaN values).")
                continue

            unique_count = len(counts_dict)
            total_count = sum(counts_dict.values())

            if unique_count > max_unique_values:
                print(f"  Too many unique values ({unique_count} distinct values). Showing top 5 most frequent:")
                items_to_show = list(counts_dict.items())[:5]
            else:
                items_to_show = list(counts_dict.items())

            for val, count in items_to_show:
                percent = (count / total_count) * 100

                print(f"  - Value '{val}': {count} occurrences ({percent:.2f}%)")

                csv_data.append({
                    'Column': column,
                    'Value': val,
                    'Count': count,
                    'Percentage (%)': round(percent, 2)
                })

        print("\n---------------------------------")

        if output_file_path:
            try:
                directory = os.path.dirname(output_file_path)

                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Created missing directory: {directory}")
                summary_df = pd.DataFrame(csv_data)
                summary_df.to_csv(output_file_path, index=False)
                print(f"\nSummary successfully saved to: {output_file_path}")
            except Exception as e:
                print(f"\nError saving summary to CSV: {e}")

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Returns the DataFrame object representing the current state of the data.
        """
        return self.df
