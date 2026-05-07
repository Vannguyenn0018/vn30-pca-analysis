import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import datetime

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Phân Tích PCA VN30 - Nhóm 1", layout="wide")

# CSS tinh chỉnh giao diện y hệt bản thiết kế
st.markdown("""
    <style>
    [data-testid="stSidebar"] {background-color: #1E3A8A; color: white;}
    [data-testid="stSidebar"] * {color: white !important;}
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; border-bottom: 2px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px;}
    .insight-box { background-color: #F0FDF4; padding: 15px; border-left: 5px solid #22C55E; border-radius: 5px; margin: 10px 0; color: #166534;}
    .breakthrough-box { background-color: #FEF2F2; padding: 15px; border-left: 5px solid #EF4444; border-radius: 5px; margin: 10px 0; color: #991B1B;}
    </style>
""", unsafe_allow_html=True)

# --- 2. LOGIC DỮ LIỆU & MAPPING NGÀNH (CHUẨN TỪ NOTEBOOK) ---
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
tickers = list(sector_mapping.keys())

# Fix triệt để lỗi "TypeError: module() takes at most 2 arguments"
@st.cache_data
def get_data():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)
    data = yf.download(tickers, start=start, end=end)['Close']
    # Điền khuyết dữ liệu 2 chiều để chắc chắn không có NaN gây lỗi PCA
    data = data.ffill().bfill() 
    return data

df_prices = get_data()
df_returns = df_prices.pct_change().dropna()

# --- TÍNH TOÁN THUẬT TOÁN PCA (GLOBAL CHO CÁC TABS) ---
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(df_returns)

cov_matrix = np.cov(scaled_returns.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Ép kiểu số thực (loại bỏ số phức) để Plotly không bị crash
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🍀 Nhóm 1 - PCA")
    st.markdown("---")
    menu = st.radio("Mục lục phân tích:", 
                    ["🏠 Tổng quan dữ liệu", "📈 Phân tích Lợi suất", "🧬 Thuật toán PCA", "🏆 Kết luận ngành"])

# --- 4. CÁC TABS NỘI DUNG ---

if menu == "🏠 Tổng quan dữ liệu":
    st.markdown('<div class="main-header">Phân Tích Cấu Trúc Thị Trường VN30</div>', unsafe_allow_html=True)
    st.write("Dữ liệu giá đóng cửa 1 năm gần nhất của 30 cổ phiếu dẫn dắt thị trường (Đã xử lý nhiễu).")
    st.dataframe(df_prices.tail(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    col1.info(f"Tổng số mẫu: {len(df_prices)} ngày giao dịch")
    col2.success(f"Số lượng mã cổ phiếu: {len(tickers)} mã")

elif menu == "📈 Phân tích Lợi suất":
    st.markdown('<div class="main-header">Phân phối lợi suất hằng ngày</div>', unsafe_allow_html=True)
    selected_stock = st.selectbox("Chọn cổ phiếu kiểm tra:", tickers)
    
    fig = px.histogram(df_returns, x=selected_stock, marginal="rug", 
                       title=f"Histogram lợi suất của mã {selected_stock}",
                       color_discrete_sequence=['#1E3A8A'])
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🧬 Thuật toán PCA":
    st.markdown('<div class="main-header">Phân tích thành phần chính (PCA)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    exp_var = eigenvalues / np.sum(eigenvalues)
    
    with col1:
        st.subheader("Giải thích phương sai (Scree Plot)")
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(10)], y=exp_var[:10]*100, name='Phương sai (%)'))
        fig_scree.update_layout(yaxis_title="% Phương sai giải thích")
        st.plotly_chart(fig_scree, use_container_width=True)
        
    with col2:
        st.subheader("Ma trận hiệp phương sai (Covariance Matrix)")
        fig_heat = px.imshow(cov_matrix[:10, :10], labels=dict(color="Covariance"),
                             x=tickers[:10], y=tickers[:10], color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f'<div class="insight-box"><b>Nhận xét:</b> Thành phần chính đầu tiên (PC1) giải thích được <b>{exp_var[0]*100:.2f}%</b> sự biến động của toàn bộ rổ VN30.</div>', unsafe_allow_html=True)

elif menu == "🏆 Kết luận ngành":
    st.markdown('<div class="main-header">Phát hiện "Sector Factors" (Nhân tố ngành)</div>', unsafe_allow_html=True)
    
    loadings = pd.DataFrame(eigenvectors[:, :3], 
                            columns=['PC1_Market', 'PC2_BDS_vs_Energy', 'PC3_Financial_Rotation'], 
                            index=tickers)
    loadings['Ngành'] = loadings.index.map(sector_mapping)
    
    sector_res = loadings.groupby('Ngành').mean(numeric_only=True)
    
    st.subheader("Bản đồ nhiệt (Heatmap) trọng số trung bình theo Ngành")
    fig_sector = px.imshow(sector_res.T, text_auto=".3f", color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_sector, use_container_width=True)
    
    st.markdown("""
    <div class="breakthrough-box">
    <b>Kết luận chuyên sâu từ mô hình PCA:</b><br>
    Việc gom cụm các hệ số Eigenvectors xác nhận có tồn tại các "Sector Factors" rõ rệt ẩn trong cấu trúc VN30:
    <ul>
        <li><b>PC1 (Nhân tố thị trường):</b> Hầu hết các ngành đều có xu hướng dịch chuyển cùng chiều, đại diện cho xu hướng chung của Vn-Index.</li>
        <li><b>PC2 (Bất động sản vs Năng lượng):</b> Thể hiện sự phân kỳ cực kỳ rõ rệt giữa nhóm Bất động sản (Dương) và Năng lượng/Công nghiệp (Âm).</li>
        <li><b>PC3 (Luân chuyển tài chính):</b> Phản ánh sự dịch chuyển dòng tiền liên quan mật thiết đến nhóm Ngân hàng/Chứng khoán so với các ngành khác.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Bảng Trọng Số PC Chi Tiết Của Từng Mã")
    st.dataframe(loadings.style.background_gradient(cmap='RdBu_r', subset=['PC1_Market', 'PC2_BDS_vs_Energy', 'PC3_Financial_Rotation']), use_container_width=True)
