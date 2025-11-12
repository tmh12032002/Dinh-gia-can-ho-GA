import pickle
import numpy as np

def tinh_fitness(model_and_scaler, du_lieu, nst):
    """
    nst: danh sách trọng số, cuối cùng là bias (intercept)
    Sử dụng scaler để transform X trước khi tính dự đoán.
    Fitness = 1 / (1 + sqrt(MSE))
    """
    model, scaler = model_and_scaler
    y_thuc_te = du_lieu.iloc[:, -1].values
    X = du_lieu.iloc[:, :-1].values

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    w = np.array(nst[:-1])
    b = float(nst[-1])

    preds = X_scaled.dot(w) + b
    mse = np.mean((y_thuc_te - preds) ** 2)
    fitness = 1.0 / (1.0 + np.sqrt(mse) + 1e-12)
    return float(fitness)

def tinh_danh_sach_fitness(model_and_scaler, du_lieu, quan_the):
    return [tinh_fitness(model_and_scaler, du_lieu, nst) for nst in quan_the]
