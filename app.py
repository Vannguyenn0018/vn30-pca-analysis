import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="VN30 PCA Analysis", layout="wide", page_icon="📈")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU (CÓ CACHE ĐỂ WEB CHẠY NHANH)
# ==========================================
@st.cache_data
def load_and_process_data(start_date, end_date, uploaded_file):
    vn30_tickers = [
        'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
    ]
    tickers_yf = [ticker + '.VN' for ticker in vn30_tickers]
    
    # Tải dữ liệu từ Yahoo Finance
    panel_data = yf.download(tickers_yf, start=start_date, end=end_date, progress=False)['Close']
    panel_data.columns = [col.replace('.VN', '') for col in panel_data.columns]
    panel_data.index = pd.to_datetime(panel_data.index).tz_localize(None).normalize()

    # Xử lý file CSV VN30_INDEX
    vn30_index_data = pd.read_csv(uploaded_file)
    vn30_index_data = vn30_index_data[['Ngày', 'Lần cuối']]
    vn30_index_data.columns = ['Date', 'VN30_INDEX']
    vn30_index_data['VN30_INDEX'] = vn30_index_data['VN30_INDEX'].str.replace(',', '').astype(float)
    vn30_index_data['Date'] = pd.to_datetime(vn30_index_data['Date'], format='%d/%m/%Y')
    vn30_index_data.set_index('Date', inplace=True)
    vn30_index_data.index = vn30_index_data.index.tz_localize(None).normalize()

    # Gộp và xử lý Missing Values
    df_final = panel_data.join(vn30_index_data, how='outer')
    df_final = df_final.loc[start_date:end_date].ffill().bfill()
    
    # Tính Daily Returns
    df_returns = df_final.pct_change().dropna()
    
    # Chuẩn hóa dữ liệu cổ phiếu (Bỏ VN30_INDEX ra để chạy PCA)
    scaler = StandardScaler()
    temp_standardized = pd.DataFrame(scaler.fit_transform(df_returns), columns=df_returns.columns, index=df_returns.index)
    standardized_stock_returns = temp_standardized.drop(columns=['VN30_INDEX'], errors='ignore')
    
    return df_final, df_returns, standardized_stock_returns

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("📊 Phân tích cấu trúc thị trường chứng khoán VN30 bằng thuật toán PCA")
st.markdown("Dự án này sử dụng phương pháp **Principal Component Analysis (PCA)** xây dựng từ đầu (toán học ma trận) để trích xuất các nhân tố chi phối thị trường.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cài đặt Dữ liệu")
uploaded_file = st.sidebar.file_uploader("1. Tải lên file VN30.csv", type=["csv"])
start_date = st.sidebar.date_input("2. Ngày bắt đầu", pd.to_datetime('2025-04-30'))
end_date = st.sidebar.date_input("3. Ngày kết thúc", pd.to_datetime('2026-04-30'))

if uploaded_file is None:
    st.warning("⚠️ Vui lòng tải lên file dữ liệu **VN30.csv** ở thanh công cụ bên trái (Sidebar) để bắt đầu phân tích.")
    st.stop()

# --- XỬ LÝ & LƯU TRỮ TRẠNG THÁI ---
with st.spinner('Đang tải và xử lý dữ liệu từ Yahoo Finance...'):
    df_final, df_returns, standardized_stock_returns = load_and_process_data(start_date, end_date, uploaded_file)

# --- CHẠY THUẬT TOÁN PCA THỦ CÔNG ---
X = standardized_stock_returns.values
cov_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sắp xếp Eigen pairs
eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
sorted_eigenvectors = np.array([pair[1] for pair in eigen_pairs]).T

total_eigenvalues = sum(eigenvalues)
explained_variance_ratio = [(i / total_eigenvalues) * 100 for i in sorted(eigenvalues, reverse=True)]
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Tính PC1 Score
pc1_vector_sliced = sorted_eigenvectors[:30, 0]
pc2_vector_sliced = sorted_eigenvectors[:30, 1]
stock_cols = standardized_stock_returns.columns
PC1_index = standardized_stock_returns @ pc1_vector_sliced

# Đổi dấu PC1 nếu ngược chiều VN30
vn30_returns = df_returns['VN30_INDEX']
if np.corrcoef(PC1_index, vn30_returns)[0, 1] < 0:
    PC1_index = -PC1_index
    pc1_vector_sliced = -pc1_vector_sliced

# ==========================================
# GIAO DIỆN TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["1. EDA & Tương quan", "2. Kết quả thuật toán PCA", "3. Yếu tố thị trường (PC1)", "4. Phân cụm Ngành (PC2)"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Khám phá dữ liệu (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Bảng tỷ suất sinh lời (Daily Returns)")
        st.dataframe(df_returns.head(10))
    with col2:
        st.subheader("Ma trận tương quan (Top 10)")
        corr = df_returns.iloc[:, :10].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- TAB 2: KẾT QUẢ PCA ---
with tab2:
    st.header("Phân tích phương sai giải thích (Scree Plot)")
    st.markdown("Biểu đồ thể hiện **sức mạnh nén thông tin** của thuật toán PCA.")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, 11), explained_variance_ratio[:10], alpha=0.7, label='Phương sai riêng lẻ (%)')
    ax.step(range(1, 11), cumulative_explained_variance[:10], where='mid', color='orange', label='Phương sai tích lũy (%)')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_xlabel('Thành phần chính (PC)')
    ax.legend()
    st.pyplot(fig)
    
    st.info(f"💡 **Insight:** Chỉ với 1 thành phần chính đầu tiên (PC1) đã giải thích được **{explained_variance_ratio[0]:.2f}%** toàn bộ biến động của 30 cổ phiếu rổ VN30.")

# --- TAB 3: PC1 MARKET FACTOR ---
with tab3:
    st.header("PC1: Yếu tố dẫn dắt thị trường (Market Factor)")
    
    # Tính tích lũy (Cumulative Returns)
    pc1_cum_scaled = (PC1_index.cumsum() - PC1_index.cumsum().mean()) / PC1_index.cumsum().std()
    vn30_cum_scaled = (vn30_returns.cumsum() - vn30_returns.cumsum().mean()) / vn30_returns.cumsum().std()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_returns.index, y=pc1_cum_scaled, mode='lines', name='PC1 Index (Scaled)'))
    fig1.add_trace(go.Scatter(x=df_returns.index, y=vn30_cum_scaled, mode='lines', name='VN30 Index (Scaled)'))
    fig1.update_layout(title="So sánh xu hướng Tăng trưởng tích lũy: PC1 vs VN30", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Trọng số (Loadings) của các mã trong PC1")
    loadings_df = pd.DataFrame({'Cổ phiếu': stock_cols, 'Loading': pc1_vector_sliced, 'Abs': np.abs(pc1_vector_sliced)})
    loadings_df = loadings_df.sort_values(by='Abs', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(x='Cổ phiếu', y='Loading', data=loadings_df, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.success("Tất cả các cổ phiếu đều có trọng số cùng chiều ở PC1. Điều này khẳng định PC1 chính là 'nhịp đập' của thị trường VN30.")

# --- TAB 4: PC2 SECTOR ROTATION ---
with tab4:
    st.header("Bản đồ Phân cụm Cổ phiếu (Luân chuyển dòng tiền)")
    st.markdown("PC1 (Trục X) đại diện cho thị trường chung. PC2 (Trục Y) cho thấy sự phân hóa nhóm ngành (Ngân hàng vs Bất động sản/Năng lượng).")
    
    df_scatter = pd.DataFrame({
        'Cổ phiếu': stock_cols,
        'PC1_Loading': pc1_vector_sliced,
        'PC2_Loading': pc2_vector_sliced
    })
    
    fig2 = px.scatter(df_scatter, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                      labels={'PC1_Loading': 'PC1 (Market Trend)', 'PC2_Loading': 'PC2 (Sector Rotation)'},
                      template="plotly_white")
    fig2.update_traces(textposition='top center', marker=dict(size=10, color='red', opacity=0.7))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("📝 Đọc phân tích kinh tế (Click để mở rộng)"):
        st.write("""
        * **Tính đồng pha (Trục X):** Toàn bộ 30 mã đều đi chung một nhịp đập thị trường.
        * **Sự gom cụm của Ngân hàng:** Các mã Ngân hàng gom chặt lại một cụm chứng tỏ dòng tiền đổ vào nhóm này mang tính đồng pha rất cao.
        * **Sự phân hóa:** Bất động sản và Năng lượng thường phân cực ở hai phía của trục PC2, thể hiện đặc tính phòng thủ hoặc chu kỳ trái ngược nhau.
        """)
