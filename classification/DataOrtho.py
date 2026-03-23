import os
import matplotlib.pyplot as plt
import seaborn as sns

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

    def generate_visualizations(self, output_dir: str = "classification/analysis/visualization") -> None:
        """
        Generates kda plots for each feature grouped by the target variable
        and a correlation heatmap.
        """
        if self.df is None or self.df.empty:
            return
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Macierz korelacji
        plt.figure(figsize=(16, 12))
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Macierz Korelacji Cech Cefalometrycznych')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
        print(f"Saved correlation heatmap to {output_dir}")


        # Wykresy gęstości dla cech z wieku 9 lat
        features_to_plot = [col for col in self.df.columns if col.startswith('9_')]
        
        fig, axes = plt.subplots(3, 5, figsize=(18, 10))
        axes = axes.flatten()
        
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
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kde_plots_9.png'))
        plt.close()
        print(f"Saved KDE plots to {output_dir}/kde_plots_9.png")

        # Wykresy gęstości dla cech z wieku 12 lat
        features_to_plot = [col for col in self.df.columns if col.startswith('12_')]
        
        fig, axes = plt.subplots(3, 5, figsize=(18, 10))
        axes = axes.flatten()
        
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

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kde_plots_12.png'))
        plt.close()
        print(f"Saved KDE plots to {output_dir}/kde_plots_12.png")