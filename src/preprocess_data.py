import pandas as pd
import yaml
import os

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_data():
    params = load_params()
    raw_path = params["data"]["raw_path"]
    processed_path = params["data"]["processed_path"]
    
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Drop missing values
    initial_shape = df.shape
    df = df.dropna(subset=['Production', 'Area'])
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values.")
    
    # Remove rows with zero Area to avoid division by zero
    df = df[df['Area'] > 0]
    
    # Calculate Yield
    df['Yield'] = df['Production'] / df['Area']
    
    # Drop original Production column as it's directly related to Yield and Area
    # We keep Area as a feature? Or drop it? 
    # Usually Yield depends on Area (diminishing return) but Yield is per unit area.
    # We'll drop Production. Area can be a feature.
    df = df.drop(columns=['Production'])
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    print(f"Saving processed data to {processed_path}...")
    df.to_csv(processed_path, index=False)
    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
