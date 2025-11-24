# Định Giá Căn Hộ Bằng Giải Thuật Di Truyền (GA)

Dự án sử dụng **Giải thuật Di truyền** để tối ưu trọng số mô hình hồi quy tuyến tính, thay vì Gradient Descent.

Mục tiêu: Dự đoán giá căn hộ dựa trên diện tích, số phòng, vị trí, tiện ích...

---

### Cách chạy

pip install pandas numpy scikit-learn matplotlib streamlit
streamlit run app.py

Các bước và Mô tả
1. Khởi tạo quần thể:Tạo ngẫu nhiên 200 cá thể (mỗi cá thể là một bộ trọng số + bias)
2. Hàm đánh giá (Fitness): Fitness = 1 / (1 + MAE) → MAE càng nhỏ → Fitness càng cao
3. Chọn lọc (Selection): Tournament Selection (k=3): chọn cá thể tốt nhất từ 3 ứng viên ngẫu nhiên
4. Lai ghép (Crossover): Single-point crossover với tỷ lệ 85%
5. Đột biến (Mutation): Mỗi gen có 10% cơ hội bị thay đổi bằng nhiễu Gaussian
6. Lặp lại: Tối đa 500 thế hệ, có Early Stopping nếu không cải thiện sau 50 thế hệ
7. Kết quả: Trả về bộ trọng số tốt nhất → dùng để dự đoán giá căn hộ mới