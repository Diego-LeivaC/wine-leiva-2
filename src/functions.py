# We are going to start preparing the data for the model

# 1st function: Split the features variables (X) with the target variable (y) and scale

from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    X = df.drop(columns=['quality', 'target'])
    y = df['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# 2nd function: Data split in training and test set

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=27):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 3rd function: Cross validation

def cross_validation(model_class, X, y, k=5, **model_params):
    kf = KFold(n_splits=k, shuffle=True, random_state=27)
    
    acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc_scores.append(accuracy_score(y_test, y_pred))
        prec_scores.append(precision_score(y_test, y_pred))
        rec_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    print(f"Cross-Validation Results ({k} folds):")
    print(f"Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Precision: {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
    print(f"Recall: {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# 4th function: Plot analysis

import matplotlib.pyplot as plt

def data_analysis(df):
    print("\nStatistic description:\n")
    print(df.describe())

    # Target variable distribution
    df['target'].value_counts().sort_index().plot(kind='bar', title='Target Variable distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Correlation matrix
    corr = df.corr(numeric_only=True)
    plt.imshow(corr, cmap='coolwarm', aspect='auto')
    plt.title("Correlation matrix")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.tight_layout()
    plt.show()

# 5th function: Plot confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix for {model_name}")
    plt.show()

# 6th funcion: Plot ROC curve
def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()