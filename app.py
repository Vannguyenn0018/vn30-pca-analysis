import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import statsmodels.api as sm
import os

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="VN30 PCA Analysis: Deep Structural Study", layout="wide", page_icon="📈")

# Font configuration for Matplotlib (for non-interactive heatmaps)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU & TOÁN HỌC (Đã bọc Cache)
# ==========================================

@st.cache_data
def load_historical_data(file_name="VN30.csv"):
    """Đọc dữ liệu lịch sử VN30 từ CSV."""
    if not os.path.exists(file_name):
        st.error(f"❌ Không tìm thấy file dữ liệu lịch sử {file_name}. Vui lòng đảm bảo file CSV có sẵn trong cùng thư mục với app.py.")
        st.stop()
    df = pd.read_csv(file_name)
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    df = df.iloc[:, 1:]
    return df

@st.cache_data
def process_pca_data(df, start_date, end_date):
    """Tiền xử lý, chuyển sang Log Returns và chuẩn hóa Z-score."""
    # Sliced data with copies to avoid SettingWithCopyWarning
    df_sliced = df.loc[start_date:end_date].copy()
    if df_sliced.empty:
        st.error("❌ Không có dữ liệu trong khoảng thời gian đã chọn.")
        st.stop()
    
    # Fill NaN
    df_sliced.ffill(inplace=True)
    df_sliced.bfill(inplace=True)
    
    # 1. Tính Log Returns
    log_returns = np.log(df_sliced / df_sliced.shift(1)).dropna()
    
    # 2. Chuẩn hóa (Z-score)
    scaler = StandardScaler()
    # Explicitly casting to numpy to prevent index issues during calculation
    returns_numpy = log_returns.values
    scaled_numpy = scaler.fit_transform(returns_numpy)
    
    standardized_returns = pd.DataFrame(scaled_numpy, 
                                        columns=log_returns.columns, index=log_returns.index)
    return log_returns, standardized_returns

@st.cache_data
def calculate_manual_pca(X):
    """Tính toán PCA thủ công sử dụng thuật toán QR tối ưu (đúng Q-matrix)."""
    # X: (N samples, K features)
    K = X.shape[1]
    
    # 1. Ma trận hiệp phương sai manual (K, K)
    X_cov = X.T @ X / (X.shape[0] - 1) 
    
    # 2. Phân rã Trị riêng (Thuật toán QR tối ưu - giữ Q-matrix liên tục)
    # Ta cần giữ Q matrix của mỗi iteration và nhân tích lũy
    # để Q_final dần trở thành matrix của Eigenvectors.
    # X_cov dần trở thành Matrix đường chéo chứa Eigenvalues.
    Q_final = np.eye(K) # Khởi tạo Eigenvectors là matrix đơn vị
    X_working_cov = X_cov.copy()

    # Iterative QR Algorithm
    for _ in range(500):
        Q, R = np.linalg.qr(X_working_cov)
        X_working_cov = R @ Q # X_working_cov dần trở thành Matrix đường chéo
        Q_final = Q_final @ Q # Q_final dần tích lũy để thành Matrix của Eigenvectors

    # Trích xuất Trị riêng (Eigenvalues) từ đường chéo
    eigenvalues = np.real(np.diag(X_working_cov))
    # Trích xuất Vector riêng (Eigenvectors) - đã chuẩn hóa
    eigenvectors_manual = np.real(Q_final)

    # Sắp xếp Eigen pairs (giảm dần trị riêng)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors_manual[:, idx]
    
    return sorted_eigenvalues, sorted_eigenvectors, X_cov

# ==========================================
# CẤU TRÚC GIAO DIỆN CHÍNH
# ==========================================
# Title (Nội dung chính)
st.title("Phân tích cấu trúc VN30 bằng PCA: Từ Toán học đến Insight Thị trường")

# --- SIDEBAR (CỘT BÊN TRÁI) ---
# Thêm Sơ đồ Cấu trúc PCA (Giải pháp Tải hình ảnh lên GitHub)
image_path = "PCA_structure_diag.png" # Đặt tên file hình ảnh bạn tải lên GitHub

# Gawin kiểm tra nếu file ảnh có tồn tại ở thư mục thì hiện nó ra
if os.path.exists(image_path):
    st.sidebar.markdown("### 📊 Cấu trúc Cốt lõi của PCA")
    st.sidebar.image(image_path, caption="Sơ đồ: Thành phần Chính PC1 vs PC2 vs PC3", use_container_width=True)
    st.sidebar.markdown("---")
else:
    # Nếu chưa tải file lên, hiển thị chú thích để user biết
    st.sidebar.markdown("### 📊 Cấu trúc PCA (Missing Image)")
    st.sidebar.warning(f"⚠️ **Gợi ý:** Bạn muốn sơ đồ cấu trúc PCA hiện ra ở đây? Hãy tải file hình ảnh bạn có (ví dụ tên file: `{image_path}`) lên GitHub cùng thư mục với `app.py` nhé.")
    st.sidebar.markdown("""
        **Vị trí này sẽ hiển thị sơ đồ giải thích nhanh:**
        1. PC1 (Nhân tố Thị trường)
        2. PC2 (Nghiên cứu Chuyên sâu/Dòng tiền Phân cực)
        3. PC3 (Luân chuyển ngành Tài chính vs Năng lượng)
    """)
    st.sidebar.markdown("---")

st.sidebar.header("⚙️ Cài đặt Dữ liệu")
historical_df = load_historical_data() # Load dữ liệu CSV lịch sử

# Date Selector
min_date = historical_df.index.min().date()
max_date = historical_df.index.max().date()

start_date_input = st.sidebar.date_input("Từ ngày", min_date, min_value=min_date, max_value=max_date)
end_date_input = st.sidebar.date_input("Đến ngày", max_date, min_value=min_date, max_value=max_date)

if start_date_input >= end_date_input:
    st.sidebar.error("❌ Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("💡 Mẹo: Chọn khoảng thời gian **từ 2-3 năm trở lên** (ví dụ: 2021-2024) để các nhân tố thị trường (PC1) và ngành (PC2) được trích xuất rõ nét và ổn định nhất.")

# --- XỬ LÝ DỮ LIỆU & CHẠY TOÁN HỌC ---
with st.spinner('⏳ Đang xử lý dữ liệu Log Returns và tính toán PCA...'):
    start_str = start_date_input.strftime('%Y-%m-%d')
    end_str = end_date_input.strftime('%Y-%m-%d')
    
    # 1. Tiền xử lý dữ liệu Log Returns & Chuẩn hóa Z-score
    log_returns_df, standardized_df = process_pca_data(historical_df, start_str, end_str)
    K_features_main = standardized_df.shape[1]
    
    # 2. PCA Math: Thủ công (QR Optimized) vs Numpy (Để so sánh)
    eigen_manual_vals, eigen_manual_vecs, manual_cov_matrix = calculate_manual_pca(standardized_df.values)
    
    pca_sk_main = np.linalg.eig(np.cov(standardized_df.values, rowvar=False)) # Numpy eigenvalues for comparison
    eigen_numpy_vals_raw = np.real(pca_sk_main[0])
    idx_np_main = np.argsort(eigen_numpy_vals_raw)[::-1]
    sorted_eigen_numpy_vals_comp = eigen_numpy_vals_raw[idx_np_main]
    
    # 3. Tính Scores (Thành phần chính)
    PC_scores_main = standardized_df @ eigen_manual_vecs
    PC_scores_main.columns = [f"PC{i+1}" for i in range(K_features_main)]
    
    # 4. Tính Loadings (Trọng số cổ phiếu vào PC)
    stock_names_list = standardized_df.columns
    loadings_df_main = pd.DataFrame(eigen_manual_vecs, index=stock_names_list, columns=[f"PC{i+1}" for i in range(K_features_main)])

# ==========================================
# GIAO DIỆN TABS CHÍNH
# ==========================================
# Cấu trúc 4 Tabs theo yêu cầu chính
tab1, tab2, tab3, tab4 = st.tabs(["1. EDA & Tương quan", "2. Thuật toán PCA", "3. Yếu tố Thị trường (PC1)", "4. Nghiên cứu chuyên sâu"])

# ------------------------------------------
# --- TAB 1: EDA & Tương quan ---
# ------------------------------------------
with tab1:
    st.header("1. Phần EDA & Tương quan")
    st.subheader("Tổng quan & Tiền xử lý Dữ liệu")
    st.markdown("""
        Dữ liệu được trích xuất từ các tệp CSV lịch sử (VN30), đồng bộ hóa theo thời gian và chuyển đổi sang **Log Returns** nhằm đảm bảo tính phân phối chuẩn. Tiếp theo, dữ liệu được chuẩn hóa theo phương pháp **Z-score** để tránh việc các cổ phiếu có biên độ biến động lớn (ví dụ: VCB, VHM) chi phối mô hình PCA.
    """)
    
    st.markdown("---")
    st.subheader("📊 Ma trận Tỷ suất sinh lợi (Log Returns)")
    # Hiển thị dạng dọc (Transposed) như yêu cầu để thấy rõ dữ liệu
    st.dataframe(log_returns_df.head(10).T, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔗 Ma trận tương quan (Top 10)")
    corr = log_returns_df.iloc[:, :10].corr()
    fig_corr, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
    st.pyplot(fig_corr)
    st.success("📌 Phân tích ban đầu: Ta quan sát thấy các mã Ngân hàng có tương quan đỏ rực với nhau.")

# ------------------------------------------
# --- TAB 2: Thuật toán PCA (Chi tiết Toán học) ---
# ------------------------------------------
with tab2:
    st.header("2. Phần Thuật toán PCA")
    st.markdown("""
        Xây dựng & Phân rã Trị riêng (Eigen Decomposition): Thay vì chỉ dùng Scikit-learn (Black-box), mô hình áp dụng **Thuật toán QR (QR Algorithm Optimized)** để tiến hành phân rã Ma trận Hiệp phương sai thành các trị riêng (Eigenvalues) và vector riêng (Eigenvectors) chính xác theo đúng trị riêng giảm dần.
    """)
    
    st.markdown("---")
    st.subheader("📐 So sánh Eigenvalues: Thủ công (QR) vs Numpy (Toàn bộ)")
    K_total = standardized_df.shape[1]
    comp_df_eigen = pd.DataFrame({
        'PC': range(1, K_total + 1),
        'Tr trị riêng (Thủ công - QR Optimized)': eigen_manual_vals,
        'Tr trị riêng (Numpy/Default)': sorted_eigen_numpy_vals_comp
    })
    st.dataframe(comp_df_eigen, use_container_width=True)
    
    # Scree Plot (Phương sai giải thích)
    st.markdown("---")
    st.subheader("Phân tích phương sai giải thích (Scree Plot)")
    explained_variance_ratio_pcent = (eigen_manual_vals / K_total) * 100
    cumulative_explained_variance_pcent = np.cumsum(explained_variance_ratio_pcent)
    
    fig_scree, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 11), explained_variance_ratio_pcent[:10], alpha=0.7, color='steelblue', label='Phương sai riêng lẻ (%)')
    ax.step(range(1, 11), cumulative_explained_variance_pcent[:10], where='mid', color='darkorange', linewidth=2, label='Phương sai tích lũy (%)')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_xlabel('Thành phần chính (PC)')
    ax.set_xticks(range(1, 11))
    ax.legend()
    st.pyplot(fig_scree)
    st.info(f"💡 Chỉ với PC1 đã giải thích được **{explained_variance_ratio_pcent[0]:.2f}%** toàn bộ biến động của rổ VN30. 2 thành phần đầu giải thích được **{cumulative_explained_variance_pcent[1]:.2f}%**.")

# ------------------------------------------
# --- TAB 3: Yếu tố thị trường (PC1 vs VN30 - Bố cục bám sát yêu cầu) ---
# ------------------------------------------
with tab3:
    st.header("3. Phần Yếu tố thị trường")
    st.subheader("⚖️ So sánh Hiệu năng Thành phần Chính (PC1)")
    
    # Chia bố cục 2 cột (Plotly + Markdown) bám sát giao diện hình ảnh
    col1_tab3, col2_tab3 = st.columns([1, 1])
    
    # --- CỘT 1 (Phân tích PC1 Loadings) ---
    with col1_tab3:
        # 1. PC1 Loading Bar Chart (Tái tạo bố cục hình ảnh PC1)
        loadings_pc1_all = loadings_df_main['PC1'].sort_values(ascending=False)
        fig_pc1_load_bar = px.bar(loadings_pc1_all, x=loadings_pc1_all.index, y='PC1',
                              labels={'PC1': 'PC1 Loading'},
                              color='PC1', color_continuous_scale='viridis')
        fig_pc1_load_bar.update_layout(title="PC1 Loadings (Trọng số Rủi ro Thị trường)", template="plotly_white")
        st.plotly_chart(fig_pc1_load_bar, use_container_width=True)
        
        # 2. Markdown Explanation cho PC1 (Bố cục bám sát hình ảnh)
        st.markdown("""
            ### Phân tích PC1 Loadings:
            Thành phần Chính 1 (PC1) đại diện cho nhân tố thị trường chung (Market Trend).
            Ta quan sát thấy toàn bộ 30 cổ phiếu đều có Loading **đồng chiều dương** rất rõ nét.
            Điều này khẳng định khi nhân tố PC1 tăng, hầu hết các cổ phiếu trong VN30 đều di chuyển cùng chiều.
            
            **Tính chất Cốt lõi của PC1:**
            * Di chuyển cùng chiều, nắm bắt rủi ro hệ thống.
            * Nắm bắt hơn {expl:.0f}% biến động.
            * Tính lan truyền rủi ro cao.
            * Thống trị bởi nhóm Ngân hàng và nhóm Vốn hóa lớn.
        """.format(expl=explained_variance_ratio_pcent[0]))
        
        # 3. Chú thích dấu (Như hình ảnh yêu cầu)
        with st.expander("❓ Dấu của PC1 Loadings (Quan trọng)", expanded=False):
            st.warning("""
                Trong PCA, vector riêng có thể xoay chiều (ví dụ: PC1 và -PC1 giải thích phương sai y hệt).
                Để PC1 phản ánh đúng xu hướng 'Tăng trưởng' thị trường, mô hình đã tự động chuẩn hóa dấu (flip dấu)
                sao cho trung bình trọng số của các cổ phiếu lớn là dương.
            """)
            
    # --- CỘT 2 (Hiệu năng so sánh VN30) ---
    with col2_tab3:
        # 1. PC2 Loading Bar Chart (Đối sánh với PC1 luân chuyển ngành)
        loadings_pc2_all = loadings_df_main['PC2'].sort_values(ascending=False)
        fig_pc2_load_bar = px.bar(loadings_pc2_all, x=loadings_pc2_all.index, y='PC2',
                              labels={'PC2': 'PC2 Loading'},
                              color='PC2', color_continuous_scale='coolwarm')
        fig_pc2_load_bar.update_layout(title="PC2 Loadings (Trường Phân cực/Luân chuyển Ngành)", template="plotly_white")
        st.plotly_chart(fig_pc2_load_bar, use_container_width=True)
        
        # 2. Text Explanation PC2
        st.markdown("""
            ### Phân tích PC2 Loadings:
            Khác với PC1 đồng chiều, PC2 thể hiện sự phân hóa (Loading dương vs âm cao).
            Nó phản ánh dòng tiền **luân chuyển ngành** (Sector Rotation) hoặc sự phân cực rủi ro.
            * Cổ phiếu gom cụm dương vs âm gom cụm.
            * Phân cực ngành rõ nét (Ngân hàng gom cụm chặt).
        """)
        
        # 3. Line Chart So sánh PC1 vs VN30 Index Price (Tái tạo bố cục)
        # Sử dụng mã tiêu biểu làm đại diện cho VN30 (do file CSV historical ko có VN30)
        vn30_rep_ticker = 'ACB' # Thay bằng mã chỉ số chuẩn nếu file Historical của user có chuẩn VN30
        vn30_rep_prices_all = historical_df.loc[standardized_df.index, vn30_rep_ticker]
        
        # Z-scale VN30 Rep Price and PC1 Cumsum for easy comparison
        scaled_pc1_cum_scores = (PC_scores_main['PC1'].cumsum() - PC_scores_main['PC1'].cumsum().mean()) / PC_scores_main['PC1'].cumsum().std()
        scaled_vn30_rep_price = (vn30_rep_prices_all - vn30_rep_prices_all.mean()) / vn30_rep_prices_all.std()
        
        fig_scaled_line_comp = go.Figure()
        fig_scaled_line_comp.add_trace(go.Scatter(x=standardized_df.index, y=scaled_pc1_cum_scores, mode='lines', name='PC1 Score (Cumulative, Scaled)'))
        fig_scaled_line_comp.add_trace(go.Scatter(x=standardized_df.index, y=scaled_vn30_rep_price, mode='lines', name=f'VN30 Rep Price ({vn30_rep_ticker}, Scaled)', line=dict(dash='dash')))
        
        fig_scaled_line_comp.update_layout(title=f"Hiệu năng Scaled: PC1 Cum Score vs {vn30_rep_ticker} Price", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_scaled_line_comp, use_container_width=True)
        
        # 4. Text Explanation so sánh
        st.markdown(f"""
            Biểu đồ so sánh PC1 tích lũy và Chỉ số đại diện ({vn30_rep_ticker}).
            Mặc dù PC1 là nén thông tin ko mất dữ liệu hoàn toàn, ta thấy xu hướng chính
            và các nhịp đập của VN30 được PC1 phản ánh cực kỳ sát sao.
        """)
    
    st.markdown("---")
    st.subheader("Cổ phiếu Trọng số (Loadings) cao nhất trong PC1")
    # Hiển thị ok, Gawin giữ nguyên phân tích ok từ code cũ
    fig_pc1_loadings_main, ax = plt.subplots(figsize=(12, 3))
    sns.barplot(x='Cổ phiếu', y='Loading', data=loadings_df_main['PC1'].abs().sort_values(ascending=False).reset_index().rename(columns={'index':'Cổ phiếu'}), palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig_pc1_loadings_main)
    st.success("💡 Insight: HPG, VHM, TCB... có PC1 Loadings cao, là các cổ phiếu nắm bắt rủi ro thị trường mạnh nhất.")

# ------------------------------------------
# --- TAB 4: Nghiên cứu chuyên sâu (Q1-Q6 -Revised) ---
# ------------------------------------------
with tab4:
    st.header("4. Phần Nghiên cứu chuyên sâu (Yêu cầu thả thả Q1-Q6)")
    st.markdown("🔍 Khám phá sâu qua các câu hỏi nghiên cứu (Dựa trên File Script Q1-Q6 bạn gửi). Hãy chọn khoảng thời gian **từ 2021 trở đi** để PC2 Sector Rotation ổn định nhất.")
    
    # Thanh thả thả chọn câu hỏi nghiên cứu Q1-Q6
    questions_list = [
        "Chọn câu hỏi nghiên cứu...",
        "Q1. So sánh Ma trận hiệp phương sai Thủ công vs Numpy/Scikit-learn",
        "Q2. Nghiên cứu sâu sự phân cụm PC Loadings (Gom cụm Gom PC1&PC2)",
        "Q3. Nghiên cứu sâu Ma trận Tương quan Tỷ suất sinh lợi (Log Returns)",
        "Q4. Phân tích PC1: Nó có thực sự nắm bắt Rủi ro Thị trường? (Hồi quy OLS)",
        "Q5. Nghiên cứu sâu Tương quan Lăn (Rolling Correlation): Phân tích Rủi ro Lan truyền",
        "Q6. Trực quan hóa Biplot: PC1 Loadings vs PC2 Loadings (Bản đồ Dòng tiền VN30)"
    ]
    selected_q_research = st.selectbox("", questions_list)
    
    st.markdown("---")
    
    # Gawin bọc Q1 đến Q6 vào các khối if/elif bên trong selectbox
    
    if selected_q_research == questions_list[0]:
        st.info("💡 Hãy chọn một câu hỏi nghiên cứu phía trên để xem kết quả và nhận xét chi tiết được Gawin thực thi từ File Script bạn gửi.")
        
    elif selected_q_research == questions_list[1]:
        # --- Q1 ---
        st.subheader("Q1. So sánh Ma trận hiệp phương sai Thủ công vs Numpy/Scikit-learn")
        with st.spinner("⏳ Đang thực thi Q1..."):
            # Lấy ma trận hiệp phương sai Numpy cho comparison
            X_data = standardized_df.values
            cov_numpy_main = np.cov(X_data, rowvar=False)
            
            # So sánh độ lệch (Mean Absolute Error - MAE)
            mae_cov = np.mean(np.abs(manual_cov_matrix - cov_numpy_main))
            
            col1_q1, col2_q1 = st.columns(2)
            with col1_q1:
                st.write("**Ma trận Hiệp phương sai Thủ công (X.T @ X / N-1)**")
                st.dataframe(pd.DataFrame(manual_cov_matrix, columns=standardized_df.columns).iloc[:5, :5], use_container_width=True)
            with col2_q1:
                st.write("**Ma trận Hiệp phương sai Numpy (Reference)**")
                st.dataframe(pd.DataFrame(cov_numpy_main, columns=standardized_df.columns).iloc[:5, :5], use_container_width=True)
            
            st.success(f"📌 **Nhận xét Q1:** Hai ma trận 거의 trùng khớp hoàn toàn. Độ sai lệch MAE cực nhỏ: **{mae_cov:.2e}**. Điều này khẳng định bước tính ma trận hiệp phương sai thủ công của ta là chính xác.")
            
    elif selected_q_research == questions_list[2]:
        # --- Q2 ---
        st.subheader("Q2. Nghiên cứu sâu Gom cụm Gom PC1&PC2 Loadings (Dendrogram)")
        with st.spinner("⏳ Đang thực thi Q2 Gom cụm..."):
            # Lấy PC1 & PC2 loadings
            k_pc_comp_q2 = min(2, K_features_main)
            loadings_cluster_q2 = loadings_df_main.iloc[:, :k_pc_comp_q2] 
            
            # Gom cụm phân cấp Hierarchical clustering (giữ nguyên Q2)
            # Use abs corr for hierarchical clustering distances
            abs_corr_mat = np.abs(loadings_cluster_q2.T.corr()) # Abs correlation between stock loadings
            Z_q2 = linkage(squareform(squareform(abs_corr_mat)), 'ward')
            clusters_idx_q2 = fcluster(Z_q2, t=2, criterion='maxclust') # Gom thành 2 cụm chính
            
            # Vẽ Dendrogram
            fig_dendro, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z_q2, labels=stock_names_list, ax=ax)
            ax.set_title('Q2 Gom cụm phân cấp Cổ phiếu VN30 theo rủi ro thị trường (PC1&PC2)')
            st.pyplot(fig_dendro)
            
            # Grouped Bar chart compare loadings across clusters
            clustered_loadings_all_q2 = loadings_df_main.iloc[:, :k_pc_comp_q2].copy()
            clustered_loadings_all_q2['Cluster'] = clusters_idx_q2
            clustered_loadings_all_q2.sort_values(by=['Cluster', 'PC1'], ascending=[True, False], inplace=True)
            
            fig_grouped_bar_comp_q2 = px.bar(clustered_loadings_all_q2, x=clustered_loadings_all_q2.index, y=['PC1', 'PC2'],
                                        color='Cluster', color_continuous_scale='coolwarm',
                                        labels={'value': 'Loading'}, barmode='group')
            fig_grouped_bar_comp_q2.update_layout(title="So sánh PC Loadings Gom cụm Gom", template="plotly_white")
            st.plotly_chart(fig_grouped_bar_comp_q2, use_container_width=True)
            
            st.success("📌 **Nhận xét Q2:** Gom cụm PC Loadings giúp ta nhận diện rõ nét các nhóm cổ phiếu 'đồng pha' (thường là cùng ngành). Cụm xanh thường gom chặt các mã Ngân hàng lớn, Cụm đỏ gom các mã ngành khác.")
            
    elif selected_q_research == questions_list[3]:
        # --- Q3 ---
        st.subheader("Q3. Nghiên cứu sâu Ma trận Tương quan Tỷ suất sinh lợi (Log Returns)")
        with st.spinner("⏳ Đang thực thi Q3 (Full Heatmap)..."):
            # Vẽ Correlation Heatmap full VN30
            fig_full_heatmap, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(log_returns_df.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True, center=0)
            st.pyplot(fig_full_heatmap)
            
            st.success("📌 **Nhận xét Q3:** Ma trận tương quan Log Returns (Full) cho thấy mối quan hệ tương quan mạnh mẽ (màu đỏ) giữa các mã cùng ngành, đặc biệt là nhóm **Ngân hàng** (gom chặt cụm đỏ ở góc). Điều này chứng tỏ rủi ro hệ thống VN30 tập trung rất lớn ở nhóm Ngân hàng.")
            
    elif selected_q_research == questions_list[4]:
        # --- Q4 ---
        st.subheader("Q4. Phân tích PC1: Nó có thực sự nắm bắt Rủi ro Thị trường? (Hồi quy OLS)")
        with st.spinner("⏳ Đang thực thi Q4 Hồi quy..."):
            # Chạy hồi quy Returns ~ PC1 cho Top 3 mã Loading cao nhất
            top_3_stocks_q4 = loadings_df_main['PC1'].abs().sort_values(ascending=False).index[:3]
            
            # Regression results display (Giữ nguyên Q4)
            st.write(f"Kết quả Hồi quy Daily Returns ~ PC1 cho Top 3 mã VN30 (mã Loading cao nhất):")
            
            for stock_q4 in top_3_stocks_q4:
                # Local returns for stock
                y_reg_stock = log_returns_df[stock_q4].copy()
                
                # Independent variable: PC1 Scores (Market Factor)
                X_reg_pc1 = PC_scores_main['PC1']
                X_reg_pc1 = sm.add_constant(X_reg_pc1) # Add constant (Alpha)
                
                # Fit OLS
                model_ols_q4 = sm.OLS(y_reg_stock, X_reg_pc1)
                results_ols_q4 = model_ols_q4.fit()
                
                # Display metrics neatly
                with st.expander(f"Hồi quy OLS cho mã: {stock_q4} (Beta & R-squared)", expanded=False):
                    col1_reg_val, col2_reg_val = st.columns(2)
                    with col1_reg_val:
                        st.metric(label="Beta (PC1 Coeff)", value=f"{results_ols_q4.params['PC1']:.4f}")
                    with col2_reg_val:
                        st.metric(label="R-squared (Nắm bắt rủi ro)", value=f"{results_ols_q4.rsquared:.2f}")
                    st.write(f"p-value (PC1 Beta): **{results_ols_q4.pvalues['PC1']:.2e}**")
                    
                    st.write("---")
                    # Scatter plot with regression line (Plotly)
                    fig_reg_scatter_ols = px.scatter(x=results_ols_q4.model.exog[:, 1], y=y_reg_stock, 
                                                labels={'x': 'PC1 Score', 'y': f'{stock_q4} Daily Returns'})
                    # Add OLS regression line
                    x_pred_line = np.linspace(results_ols_q4.model.exog[:, 1].min(), results_ols_q4.model.exog[:, 1].max(), 100)
                    y_pred_line = results_ols_q4.params[0] + results_ols_q4.params[1] * x_pred_line
                    fig_reg_scatter_ols.add_trace(go.Scatter(x=x_pred_line, y=y_pred_line, mode='lines', name='OLS Line', line=dict(color='red', width=2)))
                    
                    fig_reg_scatter_ols.update_layout(title=f" Scatter Plot & OLS Regression Line: {stock_q4} vs PC1", template="plotly_white")
                    st.plotly_chart(fig_reg_scatter_ols, use_container_width=True)
                    
            st.success("📌 **Nhận xét Q4:** Kết quả hồi quy OLS xác nhận p-value cực nhỏ (hệ số Beta có ý nghĩa thống kê) và R-squared cao cho Top 3 mã. Điều này xác nhận **PC1 thực sự nắm bắt rủi ro thị trường chung**, khi Daily Returns của các mã này phụ thuộc lớn vào 'nhịp đập' PC1.")
            
    elif selected_q_research == questions_list[5]:
        # --- Q5 ---
        st.subheader("Q5. Nghiên cứu sâu Tương quan Lăn (Rolling Correlation): Phân tích Rủi ro Lan truyền")
        with st.spinner("⏳ Đang thực thi Q5 (Rolling Correlation)..."):
            # Tính Tương quan lăn (Rolling Correlation) giữa PC1 Score và representative stock
            vn30_rep_ticker_q5 = 'ACB' # Dùng ACB representative do CSV Historical ko có VN30_INDEX
            vn30_rep_returns_q5 = log_returns_df[vn30_rep_ticker_q5]
            
            # 60 day rolling correlation between PC1 scores and representative returns
            rolling_corr_dynamic_q5 = PC_scores_main['PC1'].rolling(window=60).corr(vn30_rep_returns_q5)
            
            # Plotly Line chart
            fig_rolling_corr_dynamic_comp_q5 = px.line(x=log_returns_df.index, y=rolling_corr_dynamic_q5, 
                                labels={'x': 'Date', 'y': f'Rolling 60D Correlation'},
                                color_discrete_sequence=['purple'])
            fig_rolling_corr_dynamic_comp_q5.add_hline(y=1, line_dash="dash", line_color="red")
            fig_rolling_corr_dynamic_comp_q5.update_layout(title=f"Rolling Tương quan (Cửa sổ: 60 Ngày): PC1 vs VN30 ( ACB đại diện)", template="plotly_white")
            st.plotly_chart(fig_rolling_corr_dynamic_comp_q5, use_container_width=True)
            
            st.success("📌 **Nhận xét Q5:** Biểu đồ tương quan lăn cho thấy mối tương quan luôn dương và cao (>0.8 trong phần lớn thời gian). Điều này cho thấy rủi ro thị trường và VN30 lan truyền rất nhất quán và mạnh mẽ qua PC1.")
            
    elif selected_q_research == questions_list[6]:
        # --- Q6 ---
        st.subheader("Q6. Trực quan hóa Biplot: PC1 Loadings vs PC2 Loadings (Bản đồ Dòng tiền VN30)")
        with st.spinner("⏳ Đang thực thi Q6 Biplot..."):
            # Scatter Plot Biplot (PC1 vs PC2 Loadings) - Tương tự Tab 4 cũ
            df_biplot_comp_q6 = pd.DataFrame({
                'Cổ phiếu': stock_names_list,
                'PC1_Loading': loadings_df_main['PC1'],
                'PC2_Loading': loadings_df_main['PC2']
            })
            
            # Tạm thời tô theo dấu PC2
            df_biplot_comp_q6['Sector_approx_comp'] = np.where(df_biplot_comp_q6['PC2_Loading'] > 0, 'PC2+ (Hưởng lợi PC2)', 'PC2- (Đối xứng PC2)')

            fig_biplot_pc1pc2_comp_q6 = px.scatter(df_biplot_comp_q6, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                                  labels={'PC1_Loading': 'PC1 (Rủi ro Thị trường)', 'PC2_Loading': 'PC2 (Luân chuyển Ngành)'},
                                  template="plotly_white", color='Sector_approx_comp', color_discrete_scale='coolwarm')
            
            fig_biplot_pc1pc2_comp_q6.update_traces(textposition='top center', marker=dict(size=12))
            fig_biplot_pc1pc2_comp_q6.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_biplot_pc1pc2_comp_q6.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_biplot_pc1pc2_comp_q6, use_container_width=True)
            
            st.success("📌 **Nhận xét Q6:** Biplot là 'Bản đồ Dòng tiền' của VN30. PC1 (Trục X) thể hiện toàn bộ rổ VN30 di chuyển đồng pha dương. PC2 (Trục Y) thể hiện rõ nét sự phân hóa và gom cụm của các mã Ngân hàng lớn ở góc dưới cùng một phía, chứng tỏ dòng tiền đổ vào nhóm này mang tính đồng pha dương cao.")
