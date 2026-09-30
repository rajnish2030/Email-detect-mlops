import os
import json
import joblib
import boto3
import pandas as pd
import mlflow
import mlflow.sklearn

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


BUCKET = "email-spam-mlops-2026"
PROCESSED_KEY = "processed/2026-09-30/emails_clean.csv"

MODEL_DIR = "backend/models"
RESULTS_DIR = "backend/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

s3 = boto3.client("s3")


def fetch_data():
    obj = s3.get_object(
        Bucket=BUCKET,
        Key=PROCESSED_KEY
    )

    df = pd.read_csv(
        StringIO(
            obj["Body"].read().decode("utf-8")
        )
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nClass Distribution:")
print(df["spam"].value_counts())

X = df["text"]
y = df["spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("email-spam-detection")

models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

results = []
trained_models = {}

for model_name, algorithm in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        ("model", algorithm)
    ])

    with mlflow.start_run(run_name=model_name):

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            preds
        )

        precision = precision_score(
            y_test,
            preds,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            preds,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            preds,
            zero_division=0
        )

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "data_source",
            f"s3://{BUCKET}/{PROCESSED_KEY}"
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "tfidf_ngram_range",
            "(1,2)"
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        trained_models[model_name] = pipeline

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                preds,
                target_names=[
                    "NOT SPAM",
                    "SPAM"
                ],
                zero_division=0
            )
        )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "f1_score",
    ascending=False
)

results_df.to_csv(
    f"{RESULTS_DIR}/model_comparison.csv",
    index=False
)

best_model_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_model_name
]

model_path = f"{MODEL_DIR}/spam_classifier.pkl"

joblib.dump(
    best_model,
    model_path
)

metadata = {
    "best_model": best_model_name,
    "model_path": model_path,
    "data_source": f"s3://{BUCKET}/{PROCESSED_KEY}",
    "dataset_rows": len(df),
    "target": "spam",
    "features": "TF-IDF",
    "algorithms": [
        "Naive Bayes",
        "Logistic Regression"
    ],
    "classes": {
        "0": "NOT SPAM",
        "1": "SPAM"
    }
}

with open(
    f"{MODEL_DIR}/metadata.json",
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

print(
    f"Best Model: {best_model_name}"
)

print(
    f"Model saved: {model_path}"
)

print("\nModel Comparison:")

print(
    results_df.to_string(
        index=False
    )
)

print("\nTraining completed successfully.")
