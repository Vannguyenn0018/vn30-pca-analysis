# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="📊 Phân Tích Cấu Trúc VN30", page_icon="📈", layout="wide")

# CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 5px;}
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 20px;}
    .insight-box { background-color: #F0F9FF; padding: 15px; border-left: 5px solid #0EA5E9; border-radius: 5px; margin-bottom: 15px;}
    .breakthrough-box { background-color: #FFF1F2; padding: 15px; border-left: 5px solid #E11D48; border-radius: 5px; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Phân Tích Cấu Trúc Thị Trường VN30 Bằng PCA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ứng dụng trực quan hóa thuật toán Machine Learning trong Tài chính</div>', unsafe_allow_html=True)

# --- SIDEBAR & CACHING DỮ LIỆU ---
st.sidebar.header("⚙️ Nguồn Dữ Liệu")

# Mapping Ngành (Khớp tuyệt đối với file Notebook)
sector_mapping = {
    "ACB.VN": "Ngân hàng", "BID.VN": "Ngân hàng", "CTG.VN": "Ngân hàng",
    "HDB.VN": "Ngân hàng", "MBB.VN": "Ngân hàng", "SHB.VN": "Ngân hàng",
    "SSB.VN": "Ngân hàng", "STB.VN": "Ngân hàng", "TCB.VN": "Ngân hàng",
    "TPB.VN": "Ngân hàng", "VCB.VN": "Ngân hàng", "VIB.VN": "Ngân hàng", "VPB.VN": "Ngân hàng",
    "VHM.VN": "Bất động sản", "VIC.VN": "Bất động sản", "VRE.VN": "Bất động sản",
    "KDH.VN": "Bất động sản", "NLG.VN": "Bất động sản", "NVL.VN": "Bất động sản", "PDR.VN": "Bất động sản",
    "SSI.VN": "Chứng khoán", "BVH.VN": "Bảo hiểm",
    "FPT.VN": "Công nghệ", "MWG.VN": "Bán lẻ", "MSN.VN": "Tiêu dùng",
    "VNM.VN": "Tiêu dùng", "SAB.VN": "Tiêu dùng",
    "HPG.VN": "Vật liệu", "GVR.VN": "Công nghiệp/Cao su",
    "GAS.VN": "Năng lượng", "PLX.VN": "Năng lượng", "POW.VN": "Năng lượng"
}
vn30_tickers = list(sector_mapping.keys())

# Bỏ tham số ttl=3600 để fix triệt để lỗi TypeError trên một số phiên bản Streamlit
@st.cache_data
def load_stock_data():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    data = yf.download(vn30_tickers, start=start_date, end=end_date)['Close']
    data = data.ffill().bfill() # Xử lý 2 chiều để chặn lỗi NaN khi tính Covariance
    return data

with st.spinner('Đang tải dữ liệu 30 mã từ Yahoo Finance (1 năm gần nhất)...'):
    df_prices = load_stock_data()

# Upload file VN30-Index thực tế
st.sidebar.markdown("---")
st.sidebar.markdown("**Tải lên file VN30-Index** (Tùy chọn)")
st.sidebar.info("Upload file VN30.csv của bạn để so sánh PC1 với chỉ số thực tế.")
uploaded_file = st.sidebar.file_uploader("Chọn file VN30.csv", type=["csv"])

df_vn30_index = None
if uploaded_file is not None:
    try:
        df_vn30_index = pd.read_csv(uploaded_file, parse_dates=['Ngày'], dayfirst=True)
        df_vn30_index = df_vn30_index.set_index('Ngày').sort_index()
        if 'Lần cuối' in df_vn30_index.columns:
            df_vn30_index['VN30_Close'] = df_vn30_index['Lần cuối'].astype(str).str.replace(',', '').astype(float)
        
        df_vn30_index.index = df_vn30_index.index.tz_localize(None)
        df_prices.index = df_prices.index.tz_localize(None)
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file CSV: {e}")

# --- TIỀN XỬ LÝ & THUẬT TOÁN PCA ---
df_returns = df_prices.pct_change().dropna()

# Dùng StandardScaler chuẩn xác như trong file Notebook
scaler = StandardScaler()
df_returns_std = pd.DataFrame(scaler.fit_transform(df_returns), index=df_returns.index, columns=df_returns.columns)

cov_matrix = np.cov(df_returns_std.T)

# Ép kiểu np.real để chống văng lỗi số phức khi Plotly vẽ biểu đồ
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
cumulative_variance = np.cumsum(explained_variance_ratio)

pcs = df_returns_std.dot(eigenvectors_sorted)
pcs.columns = [f"PC{i+1}" for i in range(pcs.shape[1])]

# --- TẠO TABS HIỂN THỊ CHUẨN UI ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Tiền xử lý Dữ liệu",
    "2️⃣ Thuật toán PCA",
    "3️⃣ So sánh PC1 & VN30",
    "🚀 Nghiên cứu Chuyên sâu"
])

with tab1:
    st.markdown("### Dữ liệu Giá Đóng Cửa (YFinance)")
    st.dataframe(df_prices.tail())

    st.markdown("### Dữ liệu Lợi Suất Chuẩn Hóa")
    st.dataframe(df_returns_std.tail())

    st.markdown('<div class="insight-box"><b>Tiền xử lý:</b> Dữ liệu được thu thập từ <code>yfinance</code>. Lợi suất hằng ngày được tính toán và chuẩn hóa (StandardScaler) để đảm bảo các cổ phiếu có thị giá cao không làm lệch mô hình PCA.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Xây dựng PCA bằng Ma trận Hiệp phương sai")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ma trận Hiệp phương sai (Covariance Matrix):**")
        fig_cov = px.imshow(cov_matrix[:10, :10], x=df_prices.columns[:10], y=df_prices.columns[:10], color_continuous_scale='RdBu_r')
        fig_cov.update_layout(width=500, height=500)
        st.plotly_chart(fig_cov, use_container_width=True)

    with col2:
        st.markdown("**Biểu đồ Scree Plot:**")
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(10)], y=explained_variance_ratio[:10]*100, name='Phương sai từng PC'))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(10)], y=cumulative_variance[:10]*100, mode='lines+markers', name='Phương sai tích lũy'))
        fig_scree.update_layout(yaxis_title="% Phương sai giải thích", hovermode="x unified")
        st.plotly_chart(fig_scree, use_container_width=True)

    st.markdown(f'<div class="insight-box"><b>Nhận xét:</b> Thành phần chính đầu tiên (PC1) giải thích được <b>{explained_variance_ratio[0]*100:.2f}%</b> biến động toàn thị trường.</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Phân tích Thành Phần Chính PC1 (Yếu Tố Thị Trường)")
    col1, col2 = st.columns([1, 2])

    pc1_weights = eigenvectors_sorted[:, 0]
    if pc1_weights.mean() < 0:
        pc1_weights = -pc1_weights
        pcs['PC1'] = -pcs['PC1']

    with col1:
        st.markdown("**Trọng số cổ phiếu (PC1 Loadings):**")
        df_weights = pd.DataFrame({'Ticker': df_prices.columns, 'Weight': pc1_weights}).sort_values(by='Weight')
        fig_weights = px.bar(df_weights, x='Weight', y='Ticker
