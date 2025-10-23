# Lab 2 — SpaceX Machine Learning Prediction (Complete)
# -----------------------------------------------------
# End-to-end cells that complete all TASKS (1–12) in the ML lab.
# Assumes you installed numpy, pandas, scikit-learn, matplotlib, seaborn as in the notebook
# and loaded `data` and `X` as in the lab instructions (dataset_part_2.csv, dataset_part_3.csv).

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_confusion_matrix(y, yhat):
    cm = confusion_matrix(y, yhat)
    fig, ax = plt.subplots()
    ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.show()

# ---------- TASK 1: Create Y from data['Class'] and X standardized ----------
# Expect `data` DataFrame with 'Class' column and feature columns in `X` (from dataset_part_3.csv per lab).
Y = data['Class'].to_numpy()
# Standardize X
X_scaled = preprocessing.StandardScaler().fit_transform(X)

# ---------- TASK 3: train_test_split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)
print("Test sample size:", X_test.shape[0])  # should be 18

# ---------- TASK 4-5: Logistic Regression + GridSearchCV ----------
parameters = {'C':[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}
lr = LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)
print("LR best params:", logreg_cv.best_params_, " best val acc:", logreg_cv.best_score_)
yhat_lr = logreg_cv.predict(X_test)
print("LR test acc:", accuracy_score(Y_test, yhat_lr))
plot_confusion_matrix(Y_test, yhat_lr)

# ---------- TASK 6-7: SVM (GridSearchCV) ----------
parameters = {'kernel':['linear','rbf','sigmoid'], 'C':np.logspace(-3,3,5), 'gamma':np.logspace(-3,3,5)}
svm = SVC()
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)
print("SVM best params:", svm_cv.best_params_, " best val acc:", svm_cv.best_score_)
yhat_svm = svm_cv.predict(X_test)
print("SVM test acc:", accuracy_score(Y_test, yhat_svm))
plot_confusion_matrix(Y_test, yhat_svm)

# ---------- TASK 8-9: Decision Tree ----------
parameters = {'criterion':['gini','entropy'],
              'splitter':['best','random'],
              'max_depth': [n for n in range(1,10)],
              'max_features':['auto','sqrt'],
              'min_samples_leaf':[1,2,4],
              'min_samples_split':[2,5,10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)
print("Tree best params:", tree_cv.best_params_, " best val acc:", tree_cv.best_score_)
yhat_tree = tree_cv.predict(X_test)
acc_tree = accuracy_score(Y_test, yhat_tree)
print("Tree test acc:", acc_tree)
plot_confusion_matrix(Y_test, yhat_tree)

# ---------- TASK 10-11: KNN ----------
parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
              'algorithm':['auto','ball_tree','kd_tree','brute'],
              'p':[1,2]}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, parameters, cv=10)
knn_cv.fit(X_train, Y_train)
print("KNN best params:", knn_cv.best_params_, " best val acc:", knn_cv.best_score_)
yhat_knn = knn_cv.predict(X_test)
print("KNN test acc:", accuracy_score(Y_test, yhat_knn))
plot_confusion_matrix(Y_test, yhat_knn)

# ---------- TASK 12: Select best test performer ----------
accs = {'LogReg': accuracy_score(Y_test, yhat_lr),
        'SVM': accuracy_score(Y_test, yhat_svm),
        'DecisionTree': accuracy_score(Y_test, yhat_tree),
        'KNN': accuracy_score(Y_test, yhat_knn)}

best_model = max(accs, key=accs.get)
print("Best model on TEST set:", best_model, "with accuracy:", accs[best_model])