# End-to-End ML Pipeline for Crop Yield Prediction

This project implements an automated, reproducible Machine Learning pipeline for predicting crop yield using DVC and MLflow.

## Project Structure
- `src/`: Python source code for pipeline stages.
- `data/`: Data directory (managed by DVC).
- `models/`: Trained models (managed by DVC).
- `params.yaml`: Configuration file.
- `dvc.yaml`: DVC pipeline definition.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize the project (if not already done):
   ```bash
   git init
   dvc init
   ```

## Running the Pipeline
To execute the full pipeline (ingestion -> preprocessing -> feature engineering -> training -> evaluation):
```bash
dvc repro
```

## Tracking and Reproducibility
- **DVC**: Manages data and pipeline stages. Use `dvc dag` to see the graph.
- **MLflow**: Tracks experiments, parameters, and metrics.
  To view results:
  ```bash
  mlflow ui
  ```

## Configuration
Modify `params.yaml` to change data sources, model hyperparameters, or file paths.

## Data Source
Dataset: [Crop Yield Prediction based on Indian Agriculture](https://github.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture)
