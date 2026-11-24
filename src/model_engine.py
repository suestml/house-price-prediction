import joblib
import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "house_model.joblib")

def get_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    # Kullanıcının anlayacağı net isimlere ve gerçekçi ölçeklere çevirelim
    df.rename(columns={
        "MedInc": "Gelir_Endeksi",
        "HouseAge": "Bina_Yasi",
        "AveRooms": "Ortalama_Oda",
        "AveBedrms": "Yatak_Odasi",
        "Population": "Bolge_Nufusu",
        "AveOccup": "Hane_Kisi_Sayisi",
        "Latitude": "Enlem",
        "Longitude": "Boylam",
        "MedHouseVal": "Fiyat"
    }, inplace=True)
    # Fiyatı gerçek dolar değerine çevir (100.000$ çarpanı)
    df["Fiyat"] = df["Fiyat"] * 100000
    return df

def train_and_save():
    df = get_data()
    X = df.drop(columns=["Fiyat"])
    y = df["Fiyat"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "feature_importance": dict(zip(X.columns, model.feature_importances_))
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics, "columns": list(X.columns)}, MODEL_PATH)
    return model, metrics

def load_or_train():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        return data["model"], data["metrics"]
    return train_and_save()