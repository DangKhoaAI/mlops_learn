import os
import sys
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt

def get_wine_data():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['quality'] = data.target
    return df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main(n_epochs, eta0):
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # 1. data
    data = get_wine_data()
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # 2. scaler
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    # 3. Experiment
    mlflow.set_experiment("Wine_Quality_Prediction")

    # 4. Run
    with mlflow.start_run():
        print(f"Starting Run with n_epochs={n_epochs} and eta0={eta0}")

        # 5. Log Parameters
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("learning_rate_init", eta0)

        # 6. Iterative Training
        model = SGDRegressor(eta0=eta0, random_state=42)
        epoch_losses = [] # For plotting
        
        for epoch in range(n_epochs):
            model.partial_fit(train_x_scaled, train_y.values.ravel()) 
            
            # Calculate loss on train set
            train_pred = model.predict(train_x_scaled)
            current_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
            epoch_losses.append(current_rmse)
            
            # 7. Log Time-Series Metric
            mlflow.log_metric("train_rmse", current_rmse, step=epoch)

        print(f"Training completed after {n_epochs} epochs.")

        # 8. Create & Log Artifact (Plot)
        plt.figure(figsize=(10, 6))
        plt.plot(range(n_epochs), epoch_losses)
        plt.title("Training Loss Curve (RMSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Training RMSE")
        plt.grid(True)
        
        # Save plot to a temporary file
        plot_filename = "loss_curve.png"
        plt.savefig(plot_filename)
        plt.close() # Close plot to free memory

        # 9. Log Artifact
        mlflow.log_artifact(plot_filename)
        print(f"Logged artifact: {plot_filename}")

        # Remove temporary file
        os.remove(plot_filename)

        # 10. Final Test Evaluation
        test_pred = model.predict(test_x_scaled)
        (rmse, mae, r2) = eval_metrics(test_y, test_pred)

        # Log final test metrics
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        # 11. Log Model
        mlflow.sklearn.log_model(model, "sgd_model")

        print("--- Test Set Evaluation Results ---")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

if __name__ == "__main__":
    # Get parameters from command line
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    eta0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    
    main(n_epochs, eta0)