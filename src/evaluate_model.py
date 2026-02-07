import pandas as pd
import yaml
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def evaluate_model():
    params = load_params()
    final_path = params["data"]["final_path"]
    test_size = params["data"]["test_size"]
    random_state = params["base"]["random_state"]
    target = params["model"]["target"]
    
    print(f"Loading data from {final_path}...")
    df = pd.read_csv(final_path)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    # Reproduce the split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("Loading model...")
    model = joblib.load("models/model.pkl")
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"RMSE: {rmse}, R2: {r2}")
    
    metrics = {
        "rmse": rmse,
        "r2": r2
    }
    
    with open("scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to scores.json")

if __name__ == "__main__":
    evaluate_model()
