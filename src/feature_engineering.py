import pandas as pd
import yaml
import os
from sklearn.preprocessing import LabelEncoder
import joblib

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def feature_engineering():
    params = load_params()
    processed_path = params["data"]["processed_path"]
    final_path = params["data"]["final_path"]
    
    print(f"Loading processed data from {processed_path}...")
    df = pd.read_csv(processed_path)
    
    # Encode categorical features
    categorical_cols = ['State_Name', 'District_Name', 'Season', 'Crop']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Save encoders (optional, for inference)
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")
    
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    
    print(f"Saving featured data to {final_path}...")
    df.to_csv(final_path, index=False)
    print("Feature engineering completed.")

if __name__ == "__main__":
    feature_engineering()
