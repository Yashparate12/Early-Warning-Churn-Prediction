import os, joblib, pandas as pd
from src.preprocess import preprocess_data
from src.train import train_model

DATA_DIR = "data"
MODEL_DIR = "models"

def main():
    df = pd.read_csv("data/raw_churn.csv")
    df_clean = preprocess_data(df)
    df_clean.to_csv("data/cleaned_churn.csv", index=False)

    model, threshold = train_model(df_clean)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(threshold, "models/threshold.pkl")

if __name__ == "__main__":
    main()
