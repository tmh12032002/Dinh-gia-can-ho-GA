# app.py - GIAO DIỆN WEB ĐỊNH GIÁ CĂN HỘ BẰNG GIẢI THUẬT DI TRUYỀN
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import load_and_prepare_data
from train import train_model
from genetic_algorithm import GeneticAlgorithm

# Cấu hình trang
st.set_page_config(page_title="Định Giá Căn Hộ AI", layout="wide")
st.title("Định Giá Căn Hộ Bằng Giải Thuật Di Truyền")
st.markdown("### Dự án minh họa GA cho bài toán hồi quy định giá căn hộ")

# Sidebar
with st.sidebar:
    st.header("Cấu hình")
    run_training = st.button("Bắt đầu huấn luyện GA", type="primary")
    st.markdown("---")
    st.info("Sau khi huấn luyện xong, bạn có thể nhập thông tin căn hộ mới để dự đoán giá!")

# Khởi tạo session state để lưu kết quả
if 'best_weights' not in st.session_state:
    st.session_state.best_weights = None
    st.session_state.ga = None
    st.session_state.data = None
    st.session_state.trained = False

# Huấn luyện khi nhấn nút
if run_training or st.session_state.trained:
    with st.spinner("Đang huấn luyện mô hình bằng Giải thuật Di truyền... (khoảng 30-60 giây)"):
        data = load_and_prepare_data()
        ga, best_weights, _ = train_model()
        
        st.session_state.ga = ga
        st.session_state.best_weights = best_weights
        st.session_state.data = data
        st.session_state.trained = True

    st.success("Huấn luyện thành công!")

    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quá trình tiến hóa")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ga.best_fitness_history, label="Best Fitness", linewidth=2)
        ax.plot(ga.avg_fitness_history, label="Avg Fitness", alpha=0.7)
        ax.set_xlabel("Thế hệ")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("Độ chính xác trên tập Test")
        test_pred = ga.predict(data['X_test'], best_weights)
        mae = np.mean(np.abs(data['y_test'] - test_pred))
        r2 = np.corrcoef(data['y_test'], test_pred)[0,1]**2
        st.metric("MAE (Test)", f"{mae:.3f} tỷ")
        st.metric("R² Score", f"{r2:.3f}")

    # Trọng số tối ưu
    st.subheader("Trọng số tối ưu tìm được")
    weights_df = pd.DataFrame({
        "Đặc trưng": ['Bias'] + data['feature_names'],
        "Trọng số": best_weights.round(4)
    })
    st.dataframe(weights_df, use_container_width=True)

    # Biểu đồ dự đoán vs thực tế
    st.subheader("Dự đoán vs Giá thực tế (Test)")
    fig, ax = plt.subplots()
    ax.scatter(data['y_test'], test_pred, alpha=0.7, edgecolors='k')
    min_val = min(data['y_test'].min(), test_pred.min())
    max_val = max(data['y_test'].max(), test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Dự đoán hoàn hảo")
    ax.set_xlabel("Giá thực tế (tỷ)")
    ax.set_ylabel("Giá dự đoán (tỷ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

else:
    st.info("Nhấn nút 'Bắt đầu huấn luyện GA' ở bên trái để chạy mô hình!")
    st.stop()

# =================== DỰ ĐOÁN CĂN HỘ MỚI ===================
st.markdown("---")
st.header("Dự đoán giá căn hộ mới")

with st.form("predict_form"):
    st.write("Nhập thông tin căn hộ:")
    col1, col2 = st.columns(2)
    
    with col1:
        dien_tich = st.number_input("Diện tích (m²)", 30, 200, 70)
        phong_ngu = st.selectbox("Số phòng ngủ", [1, 2, 3, 4, 5], 2)
        phong_tam = st.selectbox("Số phòng tắm", [1, 2, 3], 1)
        noi_that = st.checkbox("Có nội thất đầy đủ", True)
        so_nam = st.number_input("Số năm sử dụng", 0, 30, 3)

    with col2:
        ban_cong = st.checkbox("Có ban công", True)
        cachtrungtam = st.slider("Cách trung tâm (km)", 1, 20, 5)
        benhvien = st.checkbox("Gần bệnh viện", True)
        truonghoc = st.checkbox("Gần trường học", True)
        muasam = st.checkbox("Gần khu mua sắm", True)

    submitted = st.form_submit_button("Định giá căn hộ")

    if submitted:
        # Chuẩn bị dữ liệu đầu vào
        new_apt = np.array([[
            dien_tich,
            phong_ngu,
            phong_tam,
            1 if noi_that else 0,
            so_nam,
            1 if ban_cong else 0,
            cachtrungtam,
            1 if benhvien else 0,
            1 if truonghoc else 0,
            1 if muasam else 0
        ]])

        # Chuẩn hóa
        scaler = st.session_state.data['scaler']
        new_apt_scaled = scaler.transform(new_apt)

        # Dự đoán
        price = st.session_state.ga.predict(new_apt_scaled, st.session_state.best_weights)[0]

        st.success(f"### GIÁ DỰ ĐOÁN: **{price:.2f} tỷ VNĐ**")
        st.balloons()

st.markdown("---")
st.caption("Dự án minh họa Giải thuật Di truyền cho bài toán định giá căn hộ")