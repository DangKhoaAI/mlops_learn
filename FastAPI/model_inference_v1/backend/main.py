import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

#app
app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#load model
try:
    model=joblib.load("../wine_model.joblib")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found. Please ensure the model is trained and the file path is correct.") 
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model=None

# Pydantic model
class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class PredictionOutput(BaseModel):
    quality: float

#API endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(features: WineFeatures):
    if model is None:
        return {"quality": None}

    input_data = np.array([[
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315_of_diluted_wines,
        features.proline
    ]])

    prediction = model.predict(input_data)
    return {"quality": prediction[0]}

#root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API"}

#run 
if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1"  ,port=8000)  
