from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

# Single day input
class WorkoutData(BaseModel):
    rolling_volume_7d: float
    rolling_volume_14d: float
    sleep_avg_7d: float
    sleep_std_7d: float
    days_since_rest: int
    acwr: float

# Batch input (list of WorkoutData)
class BatchWorkoutData(BaseModel):
    data: List[WorkoutData]

app = FastAPI(title="Recovery Risk Predictor")

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../reclaim_model.pkl")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict_risk(data: WorkoutData):
    X = np.array([[data.rolling_volume_7d, data.rolling_volume_14d, data.sleep_avg_7d,
                   data.sleep_std_7d, data.days_since_rest, data.acwr]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return {"recovery_risk": int(pred), "risk_probability": float(prob)}

@app.post("/predict_batch")
def predict_risk_batch(batch: BatchWorkoutData):
    # Convert list of WorkoutData to numpy array
    X = np.array([[d.rolling_volume_7d, d.rolling_volume_14d, d.sleep_avg_7d,
                   d.sleep_std_7d, d.days_since_rest, d.acwr] for d in batch.data])
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # probability of risk

    # Return list of results
    results = [{"recovery_risk": int(p), "risk_probability": float(prob)} 
               for p, prob in zip(preds, probs)]
    return {"results": results}