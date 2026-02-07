import pandas as pd
import yaml
import os

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def ingest_data():
    params = load_params()
    source_url = params["data"]["source_url"]
    raw_path = params["data"]["raw_path"]
    
    print(f"Downloading data from {source_url}...")
    df = pd.read_csv(source_url)
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    print(f"Saving data to {raw_path}...")
    df.to_csv(raw_path, index=False)
    print("Data ingestion completed.")

if __name__ == "__main__":
    ingest_data()
