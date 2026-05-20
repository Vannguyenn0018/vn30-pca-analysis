# **Phân tích Cấu trúc Thị trường Chứng khoán Việt Nam bằng PCA (Rổ VN30)**
--------
## **Giới thiệu dự án**
Dự án này thực hiện phân tích cấu trúc thị trường chứng khoán Việt Nam thông qua việc áp dụng **thuật toán PCA** (Principal Component Analysis) trên **30 mã cổ phiếu thuộc rổ VN30**. Mục tiêu là giải mã "nhịp đập" chung của thị trường, nhận diện các yếu tố rủi ro hệ thống và sự phân hóa giữa các nhóm ngành.

Bạn có thể xem báo cáo phân tích chi tiết tại:
👉 [Truy cập Dashboard tại đây](https://doancuoikyptdltc.streamlit.app/)

## 💡 **Các nội dung chính trong Project**
**Tiền xử lý dữ liệu:** Thu thập dữ liệu lịch sử giá từ Yahoo Finance, xử lý dữ liệu khuyết thiếu và chuẩn hóa lợi suất hằng ngày.

**PCA & Giảm chiều dữ liệu:**

* Sử dụng PCA để trích xuất các thành phần chính (Principal Components).

* **PC1:** Chứng minh là "Market Factor" (yếu tố thị trường chung) với độ tương quan lên tới ~0.93 so với VN30-Index.

* **PC2 & PC3:** Phân tích sự luân chuyển dòng tiền và sự phân hóa ngành (ví dụ: đối trọng giữa nhóm Bất động sản và Công nghiệp/Năng lượng).

* **Trực quan hóa:** Sử dụng Plotly và Seaborn để vẽ bản đồ Biplot, Heatmap tương quan và các biểu đồ trượt (Rolling Correlation) giúp quan sát sự thay đổi trạng thái thị trường theo thời gian.

🛠 **Công nghệ sử dụng**
Ngôn ngữ: Python

**Thư viện chính:** pandas, numpy, scikit-learn (PCA), yfinance, plotly, seaborn, streamlit.

📂 Cấu trúc Repository
Plaintext
├── data/               # Dữ liệu VN30 (csv)
├── notebooks/          # VN30_analysis.ipynb phân tích gốc
├── app.py              # Code triển khai Streamlit
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Hướng dẫn dự án

## 📈 **Kết luận từ dự án**
- Mô hình PCA không chỉ giúp giảm chiều dữ liệu hiệu quả mà còn cung cấp bộ lọc sắc bén để nhà đầu tư phân lớp danh mục:

- Nhóm Ngân hàng có tính đồng pha nội bộ cao (dòng tiền bầy đàn).

- Các cổ phiếu trụ cột (VIC, GAS, VJC) thể hiện đặc tính phòng thủ và độc lập hơn so với diễn biến chung.
