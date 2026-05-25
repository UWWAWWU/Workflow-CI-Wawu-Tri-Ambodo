import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# aktifkan autolog
mlflow.sklearn.autolog()

# load dataset
train_df = pd.read_csv(
    'breast_cancer_preprocessing/train.csv'
)

test_df = pd.read_csv(
    'breast_cancer_preprocessing/test.csv'
)

# split feature dan target
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# start training
with mlflow.start_run():

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    # training model
    model.fit(X_train, y_train)

    # prediksi
    y_pred = model.predict(X_test)

    # evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # logging metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("Training selesai")