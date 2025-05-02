from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the trained Random Forest model from the .pkl file
model = joblib.load("best_random_forest_model.pkl")

# Load the trained StandardScaler from the .pkl file
scaler = joblib.load("scaler.pkl")

# Define the columns to scale
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

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
    # Convert input data to a dictionary
    input_dict = input_data.dict()

    # Create a DataFrame from the input data
    df = pd.DataFrame([input_dict])

    # Preprocess only the specified columns using the scaler
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])

    # Convert the DataFrame to a NumPy array for prediction
    data = df.to_numpy()

    # Make a prediction
    prediction = model.predict(data)

    print(prediction) # to check the CI pipeline
    # Return the prediction result
    return {"prediction": int(prediction[0])}