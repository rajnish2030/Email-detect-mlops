import pandas as pd
import boto3
from io import StringIO
from datetime import date

BUCKET = "email-spam-mlops-2026"
RAW_KEY = "raw/2026-09-30/emails.csv"
CLEAN_PATH = "backend/data/emails_clean.csv"

s3 = boto3.client("s3")

obj = s3.get_object(
    Bucket=BUCKET,
    Key=RAW_KEY
)

df = pd.read_csv(
    StringIO(
        obj["Body"].read().decode("utf-8")
    )
)

print("=========== Before Cleaning ============")
print(df.isnull().sum())
print(f"Shape Before: {df.shape}")

df_clean = df.dropna(
    subset=["text", "spam"]
).copy()

df_clean["text"] = df_clean["text"].astype(str)

df_clean["spam"] = pd.to_numeric(
    df_clean["spam"],
    errors="coerce"
)

df_clean = df_clean.dropna(
    subset=["spam"]
)

df_clean["spam"] = df_clean["spam"].astype(int)

df_clean = df_clean[
    df_clean["spam"].isin([0, 1])
]

df_clean = df_clean.drop_duplicates(
    subset=["text"]
)

print("\n=========== After Cleaning ============")
print(df_clean.isnull().sum())
print(f"Shape After: {df_clean.shape}")

print("\n=========== Class Distribution ============")
print(df_clean["spam"].value_counts())

df_clean.to_csv(
    CLEAN_PATH,
    index=False
)

print(f"\nCleaned CSV saved: {CLEAN_PATH}")

def upload_processed_data(local_path):
    key = f"processed/{date.today()}/emails_clean.csv"

    s3.upload_file(
        local_path,
        BUCKET,
        key
    )

    print(
        f"\nUploaded to s3://{BUCKET}/{key}"
    )

    return key

upload_processed_data(CLEAN_PATH)