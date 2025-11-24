# data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_prepare_data(csv_path="bo_du_lieu_can_ho.csv"):
    """
    Đọc dữ liệu từ file CSV và chuẩn bị cho huấn luyện
    """
    # Kiểm tra file có tồn tại không
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {csv_path}\n"
            f"   Vui lòng đặt file 'bo_du_lieu_can_ho.csv' vào cùng thư mục với code!"
        )

    # Đọc file CSV
    print(f"Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)

    # In thông tin cơ bản
    print(f"Đã tải thành công {len(df)} căn hộ")
    print(f"Các cột: {list(df.columns)}")
    print(f"Kiểu dữ liệu:\n{df.dtypes}")
    print("-" * 60)

    # Kiểm tra có cột giá không
    if 'Gia_can_ho' not in df.columns:
        raise ValueError("File CSV phải có cột 'Gia_can_ho'!")

    # Tách đặc trưng và nhãn
    X = df.drop('Gia_can_ho', axis=1).values
    y = df['Gia_can_ho'].values

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lấy tên các đặc trưng
    feature_names = df.drop('Gia_can_ho', axis=1).columns.tolist()

    print(f"Chia dữ liệu: {len(X_train)} train | {len(X_test)} test")
    print("Chuẩn bị dữ liệu hoàn tất!\n")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'df': df,
        'raw_df': df.copy()  # giữ bản gốc nếu cần
    }


# Nếu bạn muốn chạy thử riêng file này
if __name__ == "__main__":
    data = load_and_prepare_data()
    print("5 mẫu dữ liệu đầu tiên:")
    print(data['df'].head())