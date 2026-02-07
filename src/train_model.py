import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_model():
    params = load_params()
    final_path = params["data"]["final_path"]
    test_size = params["data"]["test_size"]
    random_state = params["base"]["random_state"]
    
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    target = params["model"]["target"]
    
    print(f"Loading data from {final_path}...")
    df = pd.read_csv(final_path)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    mlflow.set_experiment(params["base"]["project"])
    
    with mlflow.start_run():
        print("Training model...")
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print(f"RMSE: {rmse}, R2: {r2}")
        
        # Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally for DVC
        joblib.dump(model, "models/model.pkl")
        print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()
