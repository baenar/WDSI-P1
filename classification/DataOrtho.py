import os
from common.DataManager import DataManager

class DataOrtho(DataManager):
    __slots__ = []

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'data', 'ortodoncja.csv')

        super().__init__(file_path=file_path)

    def clean_ortho_data(self, encode_target: bool = False) -> None:
        """
        Executes a sequence of data cleaning steps specific to the orthodontic dataset.
        """
        if self.df is None:
            return

        print("Starting orthodontic data cleaning process...")

        # DROP MISSING VALUES
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        dropped_rows = initial_rows - len(self.df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows containing missing values.")

        # ENCODE TARGET VARIABLE
        if encode_target:
            target_map = {
                'horizontal': 0,
                'normal': 1,
                'vertical': 2
            }
            self.map_values_in_column_by_dict('growth direction', target_map)
            print("Encoded target column 'growth direction' to numerical values.")

        print("Orthodontic data cleaning completed.")