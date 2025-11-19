from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model once
model = joblib.load("titanic_model.pkl")

app = FastAPI()

class Passenger(BaseModel):
    pclass: float
    sex: float
    age: float
    sibsp: float
    parch: float
    fare: float
    cabin_encoded: float
    embarked_encoded: float

@app.post("/predict")
def predict(data: Passenger):
    import traceback
    try:
        x = np.array([
            data.pclass,
            data.sex,
            data.age,
            data.sibsp,
            data.parch,
            data.fare,
            data.cabin_encoded,
            data.embarked_encoded
        ]).reshape(1, -1)

        pred = model.predict(x)[0]
        return {"survival_prediction": float(pred)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}