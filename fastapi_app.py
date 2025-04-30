from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained Random Forest model from the .pkl file
model = joblib.load("best_random_forest_model.pkl")

# Define the input data schema
class PredictionInput(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float
    sex_0: int
    sex_1: int
    cp_0: int
    cp_1: int
    cp_2: int
    cp_3: int
    fbs_0: int
    fbs_1: int
    restecg_0: int
    restecg_1: int
    restecg_2: int
    exang_0: int
    exang_1: int
    slope_0: int
    slope_1: int
    slope_2: int
    ca_0: int
    ca_1: int
    ca_2: int
    ca_3: int
    ca_4: int
    thal_0: int
    thal_1: int
    thal_2: int
    thal_3: int

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input data to a NumPy array
    data = np.array([[
        input_data.age, input_data.trestbps, input_data.chol, input_data.thalach, input_data.oldpeak,
        input_data.sex_0, input_data.sex_1, input_data.cp_0, input_data.cp_1, input_data.cp_2, input_data.cp_3,
        input_data.fbs_0, input_data.fbs_1, input_data.restecg_0, input_data.restecg_1, input_data.restecg_2,
        input_data.exang_0, input_data.exang_1, input_data.slope_0, input_data.slope_1, input_data.slope_2,
        input_data.ca_0, input_data.ca_1, input_data.ca_2, input_data.ca_3, input_data.ca_4,
        input_data.thal_0, input_data.thal_1, input_data.thal_2, input_data.thal_3
    ]])

    # Make a prediction
    prediction = model.predict(data)

    # Return the prediction result
    return {"prediction": int(prediction[0])}