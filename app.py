import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="VN30 PCA Analysis", layout="wide", page_icon="📈")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU (ĐÃ SỬA LỖI & TỐI ƯU)
# ==========================================
@st.cache_data
def load_and_process_data(start_date, end_date, uploaded_file=None):
    vn30_tickers = [
        'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
        'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
        'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE'
    ]
    tickers_yf = [ticker + '.VN' for ticker in vn30_tickers]
    
    # 1. Tải dữ liệu từ Yahoo Finance (Có bẫy lỗi)
    try:
        panel_data = yf.download(tickers_yf, start=start_date, end=end_date, progress=False)['Close']
        if panel_data.empty:
            st.error("❌ Không tải được dữ liệu từ Yahoo Finance. Vui lòng chọn khoảng thời gian khác.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Yahoo Finance: {e}")
        st.stop()

    panel_data.columns = [col.replace('.VN', '') for col in panel_data.columns]
    panel_data.index = pd.to_datetime(panel_data.index).tz_localize(None).normalize()

    # 2. Xử lý file VN30_INDEX (Ưu tiên đọc file có sẵn trên GitHub)
    try:
        if os.path.exists("VN30.csv"):
            vn30_index_data = pd.read_csv("VN30.csv")
        elif uploaded_file is not None:
            vn30_index_data = pd.read_csv(uploaded_file)
        else:
            st.warning("⚠️ Không tìm thấy file 'VN30.csv' trên hệ thống. Vui lòng tải lên ở thanh bên trái.")
            st.stop()

        vn30_index_data = vn30_index_data[['Ngày', 'Lần cuối']]
        vn30_index_data.columns = ['Date', 'VN30_INDEX']
        
        # Sửa lỗi string có dấu phẩy thành float
        if vn30_index_data['VN30_INDEX'].dtype == 'object':
            vn30_index_data['VN30_INDEX'] = vn30_index_data['VN30_INDEX'].str.replace(',', '').astype(float)
            
        vn30_index_data['Date'] = pd.to_datetime(vn30_index_data['Date'], format='%d/%m/%Y')
        vn30_index_data.set_index('Date', inplace=True)
        vn30_index_data.index = vn30_index_data.index.tz_localize(None).normalize()
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file VN30.csv: {e}")
        st.stop()

    # 3. Gộp và xử lý Missing Values
    df_final = panel_data.join(vn30_index_data, how='outer')
    df_final = df_final.loc[start_date:end_date].ffill().bfill()
    
    # 4. Tính Daily Returns & Chuẩn hóa
    df_returns = df_final.pct_change().dropna()
    scaler = StandardScaler()
    temp_standardized = pd.DataFrame(scaler.fit_transform(df_returns), columns=df_returns.columns, index=df_returns.index)
    standardized_stock_returns = temp_standardized.drop(columns=['VN30_INDEX'], errors='ignore')
    
    return df_final, df_returns, standardized_stock_returns

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("📊 Phân tích cấu trúc thị trường VN30 bằng PCA")
st.markdown("Dự án sử dụng phương pháp **Principal Component Analysis (PCA)** xây dựng từ đầu (Toán học Ma trận) để trích xuất các nhân tố chi phối dòng tiền trên thị trường chứng khoán Việt Nam.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Cài đặt Thời gian")
start_date = st.sidebar.date_input("Từ ngày", pd.to_datetime('2025-04-30'))
end_date = st.sidebar.date_input("Đến ngày", pd.to_datetime('2026-04-30'))

st.sidebar.markdown("---")
st.sidebar.markdown("*(Hệ thống sẽ tự động tìm file `VN30.csv` trên kho dữ liệu. Nếu không có, bạn có thể tải lên thủ công bên dưới)*")
uploaded_file = st.sidebar.file_uploader("Tải file VN30.csv (Tùy chọn)", type=["csv"])

# --- XỬ LÝ DỮ LIỆU ---
with st.spinner('⏳ Đang thu thập và xử lý dữ liệu...'):
    df_final, df_returns, standardized_stock_returns = load_and_process_data(start_date, end_date, uploaded_file)

# --- THUẬT TOÁN PCA ---
X = standardized_stock_returns.values
cov_matrix = np.cov(X, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eigen_pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
sorted_eigenvectors = np.array([pair[1] for pair in eigen_pairs]).T

total_eigenvalues = sum(eigenvalues)
explained_variance_ratio = [(i / total_eigenvalues) * 100 for i in sorted(eigenvalues, reverse=True)]
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

pc1_vector_sliced = sorted_eigenvectors[:30, 0]
pc2_vector_sliced = sorted_eigenvectors[:30, 1]
stock_cols = standardized_stock_returns.columns
PC1_index = standardized_stock_returns @ pc1_vector_sliced

# Đổi dấu PC1 nếu ngược chiều với VN30_INDEX
vn30_returns = df_returns['VN30_INDEX']
if np.corrcoef(PC1_index, vn30_returns)[0, 1] < 0:
    PC1_index = -PC1_index
    pc1_vector_sliced = -pc1_vector_sliced

# ==========================================
# CÁC TAB HIỂN THỊ
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["1. Tương quan Dữ liệu", "2. Sức mạnh PCA", "3. Yếu tố Thị trường (PC1)", "4. Phân hóa Dòng tiền (PC2)"])

with tab1:
    st.header("Khám phá dữ liệu & Ma trận tương quan")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Bảng tỷ suất sinh lời (Daily Returns)**")
        st.dataframe(df_returns.head(10), use_container_width=True)
    with col2:
        st.write("**Ma trận tương quan (Heatmap)**")
        corr = df_returns.iloc[:, :10].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

with tab2:
    st.header("Mức độ giải thích của các thành phần chính (Scree Plot)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 11), explained_variance_ratio[:10], alpha=0.7, color='steelblue', label='Phương sai riêng lẻ (%)')
    ax.step(range(1, 11), cumulative_explained_variance[:10], where='mid', color='darkorange', linewidth=2, label='Phương sai tích lũy (%)')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_xlabel('Thành phần chính (PC)')
    ax.legend()
    st.pyplot(fig)
    st.info(f"💡 **Insight:** Chỉ với 1 thành phần chính đầu tiên (PC1) đã giải thích được **{explained_variance_ratio[0]:.2f}%** toàn bộ biến động của 30 cổ phiếu rổ VN30. 2 thành phần đầu giải thích được **{cumulative_explained_variance[1]:.2f}%**.")

with tab3:
    st.header("PC1: Yếu tố rủi ro hệ thống (Market Factor)")
    pc1_cum_scaled = (PC1_index.cumsum() - PC1_index.cumsum().mean()) / PC1_index.cumsum().std()
    vn30_cum_scaled = (vn30_returns.cumsum() - vn30_returns.cumsum().mean()) / vn30_returns.cumsum().std()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_returns.index, y=pc1_cum_scaled, mode='lines', name='PC1 Index', line=dict(color='blue', width=2)))
    fig1.add_trace(go.Scatter(x=df_returns.index, y=vn30_cum_scaled, mode='lines', name='VN30 Index', line=dict(color='orange', width=2)))
    fig1.update_layout(title="Mức độ đồng pha giữa PC1 và Chỉ số VN30", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("**Trọng số (Loadings) của các mã trong PC1**")
    loadings_df = pd.DataFrame({'Cổ phiếu': stock_cols, 'Loading': pc1_vector_sliced, 'Abs': np.abs(pc1_vector_sliced)})
    loadings_df = loadings_df.sort_values(by='Abs', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.barplot(x='Cổ phiếu', y='Loading', data=loadings_df, palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab4:
    st.header("Bản đồ Phân cụm Cổ phiếu (PCA Biplot)")
    st.markdown("Quan sát sự luân chuyển dòng tiền giữa các nhóm ngành.")
    
    df_scatter = pd.DataFrame({'Cổ phiếu': stock_cols, 'PC1_Loading': pc1_vector_sliced, 'PC2_Loading': pc2_vector_sliced})
    
    fig2 = px.scatter(df_scatter, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                      labels={'PC1_Loading': 'PC1 (Market Trend - Xu hướng thị trường)', 'PC2_Loading': 'PC2 (Sector Rotation - Luân chuyển ngành)'},
                      template="plotly_white", size_max=12)
    fig2.update_traces(textposition='top center', marker=dict(size=10, color='crimson', opacity=0.7))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.success("📌 **Kết luận nhanh:** Trục X (PC1) cho thấy toàn bộ rổ VN30 di chuyển đồng pha với thị trường. Trục Y (PC2) cho thấy sự đối lập về dòng tiền (Ví dụ: Nhóm Bất động sản và Năng lượng thường phân cực ở hai phía).")
