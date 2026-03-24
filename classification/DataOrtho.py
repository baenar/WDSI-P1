import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from common.DataManager import DataManager

class DataOrtho(DataManager):
    __slots__ = []

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'data', 'ortodoncja.csv')

        super().__init__(file_path=file_path)

    def encode_target_variable(self) -> None:
        """
        Encodes the target variable "growth direction" into numerical values:
        """
        if self.df is None:
            return
        
        print("Encoding target variable 'growth direction'...")
        target_map = {
            'horizontal': -1,
            'normal': 0,
            'vertical': 1
        }
        self.map_values_in_column_by_dict('growth direction', target_map)
        print("Encoded target column 'growth direction' to numerical values.")

    def generate_kde_plots(
            self, 
            output_dir: str = "classification/analysis/visualization", 
            prefixes_to_plot: list = ['9_', '12_', 'delta_']
        ) -> None:
        """
        Generates KDE plots for each feature grouped by the target variable "growth direction".
        Dynamically handles different feature groups based on their prefixes (e.g., '9_', '12_', 'delta_').
        """
        if self.df is None or self.df.empty:
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for prefix in prefixes_to_plot:
            features_to_plot = [col for col in self.df.columns if col.startswith(prefix)]
            
            if not features_to_plot:
                continue
                
            cols = 5
            rows = (len(features_to_plot) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
            axes = np.array(axes).flatten() if rows > 1 else np.array(axes)
            
            for i, feature in enumerate(features_to_plot):
                sns.kdeplot(
                    data=self.df, 
                    x=feature, 
                    hue='growth direction', 
                    fill=True,         
                    common_norm=False, 
                    alpha=0.4,         
                    linewidth=2,      
                    ax=axes[i], 
                    palette='Set1'
                )
                axes[i].set_title(f'{feature}')
                axes[i].set_xlabel('')
                axes[i].set_ylabel('Gęstość')
            
            for j in range(len(features_to_plot), len(axes)):
                fig.delaxes(axes[j])
                
            plt.tight_layout()
            
            clean_prefix = prefix.rstrip('_')
            filename = f'kde_plots_{clean_prefix}.png'
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath)
            plt.close()
            print(f"Saved KDE plots to {filepath}")

    def generate_correlation_matrix(self, filename: str, output_dir: str = 'classification/analysis/visualization') -> None:
        """
        Generates a correlation heatmap for the numeric features in the dataset.
        """
        if self.df is None or self.df.empty:
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(16, 12))
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved correlation heatmap to {output_dir}")
   
    def remove_highly_correlated_features(self, threshold: float = 0.95) -> None:
        """
        Removes features that are highly correlated with each other to reduce noise.
        """
        if self.df is None:
            return
            
        print(f"Removing features with correlation > {threshold}...")
        
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        if 'growth direction' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['growth direction'])
            
        corr_matrix = numeric_df.corr().abs()
        
        import numpy as np
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            print(f"Dropped {len(to_drop)} highly correlated columns: {to_drop}")
        else:
            print("No highly correlated columns found.")

    def engineer_growth_features(self, drop_age_12: bool = True) -> None:
        """
        Calculates the delta (difference) between age 12 and age 9 measurements.
        Creates new columns with prefix 'delta_'.
        """
        if self.df is None:
            return
            
        print("Engineering new features (deltas between 9 and 12 years)...")
        base_features = [col[2:] for col in self.df.columns if col.startswith('9_')]
        
        for feature in base_features:
            col_9 = f"9_{feature}"
            col_12 = f"12_{feature}"
            
            if col_9 in self.df.columns and col_12 in self.df.columns:
                self.df[f"delta_{feature}"] = self.df[col_12] - self.df[col_9]
                
        print(f"Added {len(base_features)} new delta features.")

        if drop_age_12:
            cols_to_drop = [col for col in self.df.columns if col.startswith('12_')]
            self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            print(f"Dropped {len(cols_to_drop)} '12_' columns to prevent multicollinearity.")