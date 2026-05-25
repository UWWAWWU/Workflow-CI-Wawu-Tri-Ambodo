import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# autolog
mlflow.sklearn.autolog()

# load dataset
train_df = pd.read_csv(
    'breast_cancer_preprocessing/train.csv'
)

test_df = pd.read_csv(
    'breast_cancer_preprocessing/test.csv'
)

# split
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# parameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

# model dasar
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# mlflow run
with mlflow.start_run():

    # training tuning
    grid_search.fit(X_train, y_train)

    # best model
    best_model = grid_search.best_estimator_

    # prediksi
    y_pred = best_model.predict(X_test)

    # evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # logging manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # best parameter
    mlflow.log_params(
        grid_search.best_params_
    )

    print("Tuning selesai")