import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_wine_data():
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['quality'] = data.target
    return df

def eval_metrics(actual, pred):
    rmse =np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual,pred)
    return rmse ,mae,r2

def main():
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    #load data
    data =get_wine_data()
    #preprocessing
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    #hyperparameters
    alpha = 0.5
    l1_ratio = 0.5
    #train
    model=ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
    model.fit(train_x,train_y)

    # predict
    predictions=model.predict(test_x)

    #eval
    rmse, mae, r2 = eval_metrics(test_y, predictions)   
    print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")

    #save 
    model_path = "wine_model.joblib"
    joblib.dump(model,model_path)
    print(f"Model saved to {model_path}")

if __name__=="__main__":
    main()