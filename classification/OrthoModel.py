from sklearn import tree, ensemble, neighbors, neural_network
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class OrthoModel:
    def __init__(self):
        self.models = {
            "Decision Tree": tree.DecisionTreeClassifier(),
            "Random Forest": ensemble.RandomForestClassifier(),
            "KNN": neighbors.KNeighborsClassifier(),
            "MLP": neural_network.MLPClassifier(max_iter=1000)
        }
        pass

    def train_and_evaluate_models(self, x_train, x_test, y_train, y_test, output_dir: str = "classification/model_evaluation") -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        class_names = ["horizontal", "normal", "vertical"]
        report_path = os.path.join(output_dir, 'models_evaluation_report.txt')

        with open(report_path, 'w', encoding='utf-8') as report_file:
            i = 0
            for model_name, model in self.models.items():
                print(f"Training {model_name}...")
                model.fit(x_train, y_train.values.ravel())
                
                print(f"Evaluating {model_name}...")
                y_pred = model.predict(x_test)
                
                cm_raw = confusion_matrix(y_test, y_pred)
                cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
                
                annot_labels = np.empty_like(cm_raw, dtype=object)
                for row in range(cm_raw.shape[0]):
                    for col in range(cm_raw.shape[1]):
                        annot_labels[row, col] = f"{cm_norm[row, col]:.1%}\n({cm_raw[row, col]})"
                
                sns.heatmap(
                    cm_norm, 
                    annot=annot_labels, 
                    fmt='',          
                    cmap='Blues',          
                    vmin=0, vmax=1,      
                    ax=axes[i],
                    xticklabels=class_names, 
                    yticklabels=class_names 
                )
                
                axes[i].set_title(f'Confusion Matrix: {model_name}', fontsize=14)
                axes[i].set_xlabel('Predicted label', fontsize=12)
                axes[i].set_ylabel('True label', fontsize=12)
                
                acc = accuracy_score(y_test, y_pred)
                clf_report = classification_report(y_test, y_pred, zero_division=0)
                
                print(f"Accuracy for {model_name}: {acc:.4f}\n")
                
                report_file.write(f"========================================\n")
                report_file.write(f"Model: {model_name}\n")
                report_file.write(f"Accuracy: {acc:.4f}\n")
                report_file.write(f"Classification Report:\n{clf_report}\n\n")
                
                i += 1

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
        plt.close()
        print(f"Saved confusion matrices to {output_dir}/confusion_matrices.png")
        print(f"Saved evaluation reports to {report_path}")