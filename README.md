# **Phân tích Cấu trúc Thị trường Chứng khoán Việt Nam bằng PCA (Rổ VN30)**
--------
## **Giới thiệu dự án**
Dự án này thực hiện phân tích cấu trúc thị trường chứng khoán Việt Nam thông qua việc áp dụng **thuật toán PCA** (Principal Component Analysis) trên **30 mã cổ phiếu thuộc rổ VN30**. Mục tiêu là giải mã "nhịp đập" chung của thị trường, nhận diện các yếu tố rủi ro hệ thống và sự phân hóa giữa các nhóm ngành.

Bạn có thể xem báo cáo phân tích chi tiết tại:
👉 [Truy cập Dashboard tại đây](https://doancuoikyptdltc.streamlit.app/)

## 💡 **Quy trình thực hiện (Pipeline)**
1. **Data Acquisition:** Tự động hóa thu thập dữ liệu giá đóng cửa 1 năm (2025-2026) thông qua `yfinance` API và chuẩn hóa dữ liệu từ Investing.com.
2. **Preprocessing:** - Xử lý dữ liệu khuyết thiếu (Missing Values) bằng phương pháp Forward Fill.
   - Chuyển đổi dữ liệu sang **Daily Returns** để đảm bảo tính dừng (Stationarity) và triệt tiêu sai lệch về quy mô vốn hóa giữa các mã cổ phiếu.
3. **Machine Learning - PCA Implementation:**
   - Xây dựng ma trận hiệp phương sai (Covariance Matrix).
   - Phân tích trị riêng (Eigenvalues) và Vector riêng (Eigenvectors).
   - Trích xuất 3 thành phần chính (PCs) quan trọng nhất.
4. **Insight Extraction:** - Phân tích tương quan giữa PC1 và VN30-Index.
   - Biplot visualization để phân cụm ngành.
   - Rolling Correlation để bắt trọn các cú sốc thị trường ngắn hạn.
   - 
## 📊 Kết quả đạt được
* **Market Factor (PC1):** Giải thích gần **39.39%** tổng phương sai. Hệ số tương quan lên đến **0.93** với chỉ số VN30-Index, khẳng định PC1 là "nhịp đập" của thị trường.
* **Sector Leadership:** Xác định rõ sự dẫn dắt của nhóm Ngân hàng (TPB, ACB, TCB) đối với xu hướng chung.
* **Sector Rotation (PC2/PC3):** Phát hiện sự phân hóa dòng tiền giữa nhóm Bất động sản và nhóm Công nghiệp/Năng lượng.

## 🛠 **Công nghệ sử dụng**
- **Ngôn ngữ:** Python
- **Phân tích dữ liệu:** `pandas`, `numpy`, `scikit-learn` (PCA)
- **Trực quan hóa:** `plotly`, `seaborn`, `matplotlib`
- **Triển khai ứng dụng:** `streamlit`
- **Dữ liệu:** `yfinance`

## **📂 Cấu trúc Repository**
```text
├── data/               # Dữ liệu VN30 (csv)
├── notebooks/          # VN30_analysis.ipynb phân tích gốc
├── app.py              # Code triển khai Streamlit
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Hướng dẫn dự án

## 📈 **Kết luận từ dự án**
- Mô hình PCA không chỉ giúp giảm chiều dữ liệu hiệu quả mà còn cung cấp bộ lọc sắc bén để nhà đầu tư phân lớp danh mục:

- Nhóm Ngân hàng có tính đồng pha nội bộ cao (dòng tiền bầy đàn).

- Các cổ phiếu trụ cột (VIC, GAS, VJC) thể hiện đặc tính phòng thủ và độc lập hơn so với diễn biến chung.
