# main.py
from data import load_and_prepare_data
from train import train_model
from evaluate import evaluate_model
from visualize import plot_results
from predict import predict_new_apartment

if __name__ == "__main__":
    print("="*60)
    print("DỰ ÁN: ĐỊNH GIÁ CĂN HỘ BẰNG GIẢI THUẬT DI TRUYỀN")
    print("="*60)

    # Load dữ liệu
    data = load_and_prepare_data()

    # Huấn luyện
    ga, best_weights, data = train_model()

    # Đánh giá
    test_pred = evaluate_model(ga, best_weights, data)

    # Hiển thị trọng số
    print("\nTRỌNG SỐ TỐI ƯU:")
    for name, w in zip(['Bias'] + data['feature_names'], best_weights):
        print(f"{name:20s}: {w:8.4f}")

    # Vẽ đồ thị
    plot_results(ga, data, test_pred, best_weights)

    # Dự đoán mẫu
    predict_new_apartment(data['scaler'], ga, best_weights, data['feature_names'])

    print("\nHOÀN THÀNH! Kết quả đã lưu: ket_qua_dinh_gia_can_ho.png")