from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model
model = joblib.load("iris_model.joblib")

# Create API with FastAPI
app = FastAPI()

# Define input schema
class Features(BaseModel):
    features: list

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Simple ML Inference API is running."}

# Inference endpoint
@app.post("/predict")
def predict(data: Features):
    # Convert input to numpy array
    input_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
