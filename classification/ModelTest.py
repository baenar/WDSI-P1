import pandas as pd
from classification.OrthoModel import OrthoModel

x_train = pd.read_csv("classification/data/default_X_train.csv")
y_train = pd.read_csv("classification/data/default_y_train.csv")
x_test = pd.read_csv("classification/data/default_X_test.csv")
y_test = pd.read_csv("classification/data/default_y_test.csv")

model = OrthoModel()
model.train_and_evaluate_models(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)