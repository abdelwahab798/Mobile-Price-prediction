from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(title="Smartphone Price Predictor")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # بيسمح لأي مكان يكلم الـ API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path= r"C:\Users\nice\Desktop\Big Data project\Mobile-Price-prediction\model\full_pipline.pkl"
model=joblib.load(model_path)

class MobileSpecs(BaseModel):
    brand: str
    chipset: str
    display_type: str
    ram_gb: int
    battery_mah: int
    cpu_score: int
    gpu_score: int
    rear_camera_mp: int
    refresh_rate_hz: int
    expandable_storage: int

@app.get("/")
def home():
    return {"message":"Welcome to Mobile Price Prediction API"}

@app.post("/predict")
def predict(specs: MobileSpecs):
    input_df = pd.DataFrame([specs.dict()])
    prediction = model.predict(input_df)
    return {"predicted_price": round(float(prediction[0]), 2)}