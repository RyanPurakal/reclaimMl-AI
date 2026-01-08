from src.ingest import generate_data
from src.label import label_risk
from src.train import train_model

df = generate_data()
df = label_risk(df)

model, X_test, y_test = train_model(df)
print("Training successful! Model:", model)

import joblib
joblib.dump(model, "reclaim_model.pkl")
print("Model saved to reclaim_model.pkl")