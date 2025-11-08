import os
import sys
import warnings
import pandas as pd
import numpy as np

# Import thư viện Scikit-learn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_wine

# Import MLflow
import mlflow
import mlflow.sklearn

# Data
def get_wine_data():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['quality'] = data.target  # Sử dụng 'target' làm biến mục tiêu
    return df

# Eval
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main(alpha, l1_ratio):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # 1.data
    data = get_wine_data()

    # 2. preprocess
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # 3. Name Experiment 
    mlflow.set_experiment("Wine_Quality_Prediction")

    # 4. Run with MLflow
    with mlflow.start_run():
        print(f"Bắt đầu Run với alpha={alpha} và l1_ratio={l1_ratio}")

        # 5.Train
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        # 6.Predict
        predictions = model.predict(test_x)

        # 7.Eval
        (rmse, mae, r2) = eval_metrics(test_y, predictions)

        # 8. Log
        
        # Log Parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log Model
        mlflow.sklearn.log_model(model, "model")

        print("--- Eval Result ---")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")
        print("log model and  metrics to MLflow.")

if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    main(alpha, l1_ratio)