import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

def huan_luyen_mo_hinh(duong_dan_csv="data/bo_du_lieu_can_ho.csv"):
    data = pd.read_csv(duong_dan_csv)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Lưu model và scaler
    with open("ml_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print("✅ Đã huấn luyện và lưu model.")
    print(f"   R² train: {model.score(X_scaled, y):.4f}")

def load_model_and_scaler(pickle_path="ml_model.pkl"):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "model" in data and "scaler" in data:
        return data["model"], data["scaler"]
    # fallback: old format (model only)
    return data, None

def du_doan(model, he_so):
    """
    Dự đoán giá trị (Price) dựa vào hệ số của NST
    """
    import numpy as np
    return float(np.dot(model.coef_, he_so) + model.intercept_)
