import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression

# mengaktifkan autolog
mlflow.sklearn.autolog()

# load dataset
train_df = pd.read_csv('breast_cancer_preprocessing/train.csv')
test_df = pd.read_csv('breast_cancer_preprocessing/test.csv')

# split feature dan target
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

with mlflow.start_run():
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print("Training selesai")
    print(f"Accuracy: {score}")