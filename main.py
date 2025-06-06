import pandas as pd
import numpy as np
from src.functions import preprocess_data, split_data, cross_validation
from src.logistic_regression import LogisticRegression
from src.svm import SVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

red_data = pd.read_csv("data/winequality-red.csv", sep=";")
white_data = pd.read_csv("data/winequality-white.csv", sep=";")

df = pd.concat([red_data, white_data], ignore_index=True)
df['target'] = np.where(df['quality'] >= 6, 1, 0)

X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

log = LogisticRegression(learning_rate=0.015, n_iters=1500)
log.fit(X_train, y_train)
y_pred_log = log.predict(X_test)

svm = SVM(learning_rate=0.01, lambda_param=0.015, n_iters=1500)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

def evaluate(y_true, y_pred, model_name):
    print(f"\nEvaluation for {model_name}:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

evaluate(y_test, y_pred_log, "Logistic Regression")
evaluate(y_test, y_pred_svm, "SVM")

print("Cross validation for Logistic Regression")
cross_validation(LogisticRegression, X, y, k=5, learning_rate=0.015, n_iters=1500)

print("Validación cruzada para SVM")
cross_validation(SVM, X, y, k=5, learning_rate=0.01, lambda_param=0.015, n_iters=1500)