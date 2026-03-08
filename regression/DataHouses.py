import os
import pandas as pd
from sympy.logic.boolalg import Boolean

from common.DataManager import DataManager


class DataHouses(DataManager):
    __slots__ = []

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'data', 'domy.csv')

        super().__init__(file_path=file_path)

    def clean_houses_data(self, do_encode_categorical_features: bool = False) -> None:
        """
        Executes a sequence of data cleaning steps specific to the houses dataset.
        """
        if self.df is None:
            return

        print("Starting data cleaning process...")

        # DROPPING USELESS COLUMNS
        columns_to_drop = [
            'Id',
            'Alley',
            'PoolQC',
            'Fence',
            'MiscFeature'
        ]
        for col in columns_to_drop:
            self.remove_column_from_set(col)
            print(f"Dropped column: {col}")

        # REPLACING '?' WITH None
        categorical_cols_with_na = [
            'GarageType',
            'GarageFinish',
            'GarageQual',
            'GarageCond',
            'BsmtQual',
            'BsmtCond',
            'BsmtExposure',
            'BsmtFinType1',
            'BsmtFinType2',
            'MasVnrType',
            'FireplaceQu'
        ]
        for col in categorical_cols_with_na:
            self.replace_value_in_column(col, '?', 'None')

        # REPLACING '?' WITH 0
        numerical_cols_with_na = [
            'MasVnrArea',
            'LotFrontage',
            'GarageYrBlt'
        ]
        for col in numerical_cols_with_na:
            self.replace_value_in_column(col, '?', '0')
            self.df[col] = pd.to_numeric(self.df[col])

        # REPLACE WORD QUALITY WITH NUMERICAL VALUE
        quality_map = {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            'None': 0,
            '?': 0
        }
        quality_columns = [
            'ExterQual',
            'ExterCond',
            'BsmtQual',
            'BsmtCond',
            'HeatingQC',
            'KitchenQual',
            'FireplaceQu',
            'GarageQual',
            'GarageCond'
        ]
        for col in quality_columns:
            self.map_values_in_column_by_dict(col, quality_map)

        if do_encode_categorical_features:
            columns_to_encode = self.get_categorical_columns_names()
            print(f"Encoding {len(columns_to_encode)} columns...")
            for col in columns_to_encode:
                self.encode_categorical_feature(col)
                print(f"Encoded column: {col}")

        print("House data cleaning completed.")
