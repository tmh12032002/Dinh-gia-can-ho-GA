# predict.py
import numpy as np

def predict_new_apartment(scaler, ga, best_weights, feature_names):
    print("\n" + "="*60)
    print("DỰ ĐOÁN CĂN HỘ MỚI")
    print("="*60)

    apt = np.array([[70, 2, 1, 1, 3, 1, 5, 1, 1, 1]])  # mẫu
    apt_scaled = scaler.transform(apt)
    price = ga.predict(apt_scaled, best_weights)[0]

    print("Căn hộ: 70m², 2PN, đầy đủ tiện ích, cách trung tâm 5km")
    print(f"→ GIÁ DỰ ĐOÁN: {price:.2f} tỷ VNĐ")