from sklearn import tree, ensemble, neighbors, neural_network
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class OrthoModel:
    def __init__(self, random_state):
        self.models = {
            "Decision Tree": tree.DecisionTreeClassifier(random_state=random_state),
            "Random Forest": ensemble.RandomForestClassifier(random_state=random_state),
            "KNN": neighbors.KNeighborsClassifier(),
            "MLP": neural_network.MLPClassifier(max_iter=3000, random_state=random_state)
        }
        pass

    def train_and_evaluate_models(
            self, X, y, 
            random_state, 
            output_dir: str = "classification/model_evaluation",
            report_filename: str = "cv_report.txt",
            confusion_matrices_filename: str = "cv_confusion_matrices.png"
        ) -> None:
        """
        Generates a comprehensive final report using the entire dataset.
        Uses RandomizedSearchCV/GridSearchCV for tuning, and cross_val_predict 
        to generate overall confusion matrices.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_path = os.path.join(output_dir, report_filename)
        class_names = ["horizontal", "normal", "vertical"]
        
        # Definicje parametrów
        param_spaces = {
            "Decision Tree": {
                'model__max_depth': [None, 5, 10, 15],
                'model__min_samples_split': [2, 5, 10],
                'model__class_weight': ['balanced', None]
            },
            "Random Forest": {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 30],
                'model__class_weight': ['balanced', 'balanced_subsample']
            },
            "KNN": {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            },
            "MLP": {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'model__activation': ['relu', 'tanh'],
                'model__alpha': [0.0001, 0.001, 0.01]
            }
        }

        # Odtwarzalna kroswalidacja
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        with open(report_path, 'w', encoding='utf-8') as report_file:
            for i, (model_name, base_model) in enumerate(self.models.items()):
                print(f"Starting tuning for model: {model_name}...")
                
                pipeline = ImbPipeline([
                    ('scaler', StandardScaler()),
                    ('smote', SMOTE(random_state=random_state)),
                    ('model', base_model)
                ])

                param_space = param_spaces.get(model_name, {})

                # Wybór optymalizatora: RandomizedSearchCV dla dużych przestrzeni, Grid dla małych
                if model_name in ["Random Forest", "MLP"]:
                    search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=param_space,
                        n_iter=10,     
                        cv=cv,
                        scoring='f1_macro',
                        random_state=random_state, 
                        n_jobs=-1
                    )
                else:
                    search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_space,
                        cv=cv,
                        scoring='f1_macro',
                        n_jobs=-1
                    )

                search.fit(X, y.values.ravel())
                best_pipeline = search.best_estimator_

                # Generowanie predykcji z kroswalidacji dla macierzy konfuzji
                print(f"Generating CV predictions (cross_val_predict) for: {model_name}...")
                y_pred_cv = cross_val_predict(best_pipeline, X, y.values.ravel(), cv=cv)

                # Rysowanie Macierzy Konfuzji
                cm_raw = confusion_matrix(y, y_pred_cv)
                cm_norm = confusion_matrix(y, y_pred_cv, normalize='true')
                
                annot_labels = np.empty_like(cm_raw, dtype=object)
                for row in range(cm_raw.shape[0]):
                    for col in range(cm_raw.shape[1]):
                        annot_labels[row, col] = f"{cm_norm[row, col]:.1%}\n({cm_raw[row, col]})"
                
                sns.heatmap(
                    cm_norm, annot=annot_labels, fmt='', cmap='Blues', 
                    vmin=0, vmax=1, ax=axes[i], xticklabels=class_names, yticklabels=class_names 
                )
                axes[i].set_title(f'CV Confusion Matrix: {model_name}', fontsize=14)
                axes[i].set_xlabel('Predicted label', fontsize=12)
                axes[i].set_ylabel('True label', fontsize=12)

                # Zapisywanie statystyk do raportu
                clf_report = classification_report(y, y_pred_cv, zero_division=0)
                
                report_file.write(f"========================================\n")
                report_file.write(f"Model: {model_name}\n")
                report_file.write(f"Optimizer: {'RandomizedSearchCV' if model_name in ['Random Forest', 'MLP'] else 'GridSearchCV'}\n")
                report_file.write(f"Best parameters:\n{search.best_params_}\n")
                report_file.write(f"Best score (F1 Macro during search): {search.best_score_:.4f}\n")
                report_file.write(f"\nOverall CV report (cross_val_predict):\n{clf_report}\n\n")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, confusion_matrices_filename))
        plt.close()
        
        print(f"\nProcess completed. Report and matrices saved to directory: {output_dir}")