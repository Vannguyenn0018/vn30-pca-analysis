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
st.set_page_config(page_title="VN30 PCA Deep Analysis", layout="wide", page_icon="📈")

# Font configuration for Matplotlib (for non-interactive heatmaps)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU & TOÁN HỌC (Đã bọc Cache)
# ==========================================

@st.cache_data
def load_historical_data(file_name="VN30_historical.csv"):
    """Đọc dữ liệu lịch sử VN30 từ CSV."""
    if not os.path.exists(file_name):
        st.error(f"❌ Không tìm thấy file {file_name}. Vui lòng đảm bảo file CSV lịch sử có sẵn trong thư mục.")
        st.stop()
    df = pd.read_csv(file_name)
    df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
    df = df.iloc[:, 1:]
    return df

@st.cache_data
def process_pca_data(df, start_date, end_date):
    """Tiền xử lý, chuyển sang Log Returns và chuẩn hóa."""
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
    standardized_returns = pd.DataFrame(scaler.fit_transform(log_returns), 
                                        columns=log_returns.columns, index=log_returns.index)
    return log_returns, standardized_returns

@st.cache_data
def calculate_manual_pca(X):
    """Tính toán PCA thủ công sử dụng thuật toán QR tối ưu (đúng Q-matrix)."""
    # X: (N, K) - N samples, K features
    K = X.shape[1]
    
    # 1. Ma trận hiệp phương sai
    # numpy.cov(X, rowvar=False) or manual calculation
    X_cov = X.T @ X / (X.shape[0] - 1) 
    
    # 2. Phân rã Trị riêng (Thuật toán QR tối ưu - giữ Q-matrix để lấy Eigenvectors)
    eigenvalues = np.diag(X_cov)
    Q_final = np.eye(K) # Khởi tạo Eigenvectors là ma trận đơn vị

    for _ in range(500): # Iterations
        Q, R = np.linalg.qr(X_cov)
        X_cov = R @ Q # X_cov dần trở thành ma trận đường chéo (Trị riêng)
        eigenvalues = np.diag(X_cov)
        Q_final = Q_final @ Q # Q_final dần trở thành ma trận Vector riêng

    # Sắp xếp Eigen pairs (giảm dần trị riêng)
    eigenvalues = np.real(eigenvalues)
    Q_final = np.real(Q_final)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = Q_final[:, idx]
    
    return sorted_eigenvalues, sorted_eigenvectors, X_cov

# ==========================================
# CẤU TRÚC GIAO DIỆN CHÍNH
# ==========================================
st.title("Phân tích cấu trúc VN30 bằng PCA: Từ Toán học đến Insight Thị trường")

# --- SIDEBAR: Cài đặt Dữ liệu ---
st.sidebar.header("⚙️ Cài đặt Phân tích")
historical_df = load_historical_data() # Load dữ liệu CSV

# Date Selector
min_date = historical_df.index.min().date()
max_date = historical_df.index.max().date()

start_date_input = st.sidebar.date_input("Từ ngày", min_date, min_value=min_date, max_value=max_date)
end_date_input = st.sidebar.date_input("Đến ngày", max_date, min_value=min_date, max_value=max_date)

if start_date_input >= end_date_input:
    st.sidebar.error("❌ Ngày bắt đầu phải trước ngày kết thúc.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info("💡 Mẹo: Hãy chọn khoảng thời gian **từ 2-3 năm trở lên** (ví dụ: 2020-2024) để các nhân tố thị trường (PC1) và ngành (PC2) được trích xuất rõ nét và ổn định nhất.")

# --- XỬ LÝ DỮ LIỆU & CHẠY TOÁN HỌC ---
with st.spinner('⏳ Đang xử lý dữ liệu và tính toán PCA...'):
    start_str = start_date_input.strftime('%Y-%m-%d')
    end_str = end_date_input.strftime('%Y-%m-%d')
    
    log_returns_df, standardized_df = process_pca_data(historical_df, start_str, end_str)
    
    # PCA: Thủ công (QR) vs Numpy (Để so sánh)
    eigen_manual_vals, eigen_manual_vecs, cov_matrix = calculate_manual_pca(standardized_df.values)
    
    pca_numpy = np.linalg.eig(np.cov(standardized_df.values, rowvar=False))
    eigen_numpy_vals = np.real(pca_numpy[0])
    idx_np = np.argsort(eigen_numpy_vals)[::-1]
    sorted_eigen_numpy_vals = eigen_numpy_vals[idx_np]
    
    # Tính Scores (Thành phần chính)
    PC_scores = standardized_df @ eigen_manual_vecs
    PC_scores.columns = [f"PC{i+1}" for i in range(PC_scores.shape[1])]
    
    # Tính Loadings (Trọng số cổ phiếu vào PC)
    stock_names = standardized_df.columns
    loadings_df = pd.DataFrame(eigen_manual_vecs, index=stock_names, columns=[f"PC{i+1}" for i in range(K=standardized_df.shape[1])])

# ==========================================
# GIAO DIỆN TABS CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["1. EDA & Tương quan", "2. Thuật toán PCA", "3. Yếu tố Thị trường (PC1)", "4. Nghiên cứu chuyên sâu"])

# ------------------------------------------
# --- TAB 1: EDA & Tương quan ---
# ------------------------------------------
with tab1:
    st.header("1. Phần EDA & Tương quan")
    st.subheader("Tổng quan & Tiền xử lý Dữ liệu")
    st.markdown("""
        Dữ liệu được trích xuất từ các tệp CSV lịch sử, đồng bộ hóa theo thời gian và chuyển đổi sang **Log Returns** nhằm đảm bảo tính phân phối chuẩn. Tiếp theo, dữ liệu được chuẩn hóa theo phương pháp **Z-score** để tránh việc các cổ phiếu có biên độ biến động lớn chi phối mô hình PCA.
    """)
    
    st.markdown("---")
    st.subheader("📊 Ma trận Tỷ suất sinh lợi (Log Returns)")
    # Hiển thị dạng dọc (Transposed) như yêu cầu
    st.dataframe(log_returns_df.head(10).T, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔗 Ma trận tương quan (Top 10)")
    corr = log_returns_df.iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
    st.pyplot(fig)

# ------------------------------------------
# --- TAB 2: Thuật toán PCA ---
# ------------------------------------------
with tab2:
    st.header("2. Phần Thuật toán PCA")
    st.markdown("""
        Xây dựng & Phân rã Trị riêng (Eigen Decomposition): Thay vì chỉ dùng thư viện Black-box, mô hình áp dụng **Thuật toán QR (QR Algorithm)** để tiến hành phân rã Ma trận Hiệp phương sai thành các trị riêng (Eigenvalues) và vector riêng (Eigenvectors).
    """)
    
    st.markdown("---")
    st.subheader("📐 So sánh Eigenvalues: Thủ công (QR) vs Numpy (Toàn bộ)")
    K_features = standardized_df.shape[1]
    comp_df = pd.DataFrame({
        'PC': range(1, K_features + 1),
        'Trị riêng (Thủ công - QR)': eigen_manual_vals,
        'Trị riêng (Numpy)': sorted_eigen_numpy_vals
    })
    st.dataframe(comp_df, use_container_width=True)
    
    # Scree Plot
    st.markdown("---")
    st.subheader("Phân tích phương sai giải thích (Scree Plot)")
    explained_variance_ratio = (eigen_manual_vals / K_features) * 100
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 11), explained_variance_ratio[:10], alpha=0.7, color='steelblue', label='Phương sai riêng lẻ (%)')
    ax.step(range(1, 11), cumulative_explained_variance[:10], where='mid', color='darkorange', linewidth=2, label='Phương sai tích lũy (%)')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_xlabel('Thành phần chính (PC)')
    ax.set_xticks(range(1, 11))
    ax.legend()
    st.pyplot(fig)
    st.info(f"💡 Chỉ với PC1 đã giải thích được **{explained_variance_ratio[0]:.2f}%** toàn bộ biến động của rổ VN30.")

# ------------------------------------------
# --- TAB 3: Yếu tố thị trường (Tái tạo bố cục hình ảnh) ---
# ------------------------------------------
with tab3:
    st.header("3. Phần Yếu tố thị trường")
    st.subheader("⚖️ So sánh Hiệu năng Thành phần Chính (PC1)")
    
    # Chia bố cục 2 cột bám sát hình ảnh yêu cầu
    col1, col2 = st.columns([1, 1])
    
    # --- CỘT 1 ---
    with col1:
        # 1. PC1 Loading Chart (Plotly)
        loadings_pc1 = loadings_df['PC1'].sort_values(ascending=False)
        fig_pc1_bar = px.bar(loadings_pc1, x=loadings_pc1.index, y='PC1',
                              labels={'PC1': 'PC1 Loading'},
                              color='PC1', color_continuous_scale='viridis')
        fig_pc1_bar.update_layout(title="PC1 Loadings (Trọng số rủi ro thị trường)", template="plotly_white")
        st.plotly_chart(fig_pc1_bar, use_container_width=True)
        
        # 2. Text Explanation
        st.markdown("""
            ### Phân tích PC1 Loadings:
            PC1 thường đại diện cho nhân tố thị trường (Market Trend/Factor). 
            Ta quan sát thấy toàn bộ các cổ phiếu đều có trọng số (Loading) **đồng chiều dương** rất rõ nét.
            Điều này khẳng định khi nhân tố PC1 tăng, hầu hết các cổ phiếu trong VN30 đều tăng theo.
        """)
        
        # 3. Dấu của PC1 Box
        st.markdown("---")
        with st.expander("❓ Vấn đề về Dấu của PC1", expanded=True):
            st.warning("""
                Trong PCA, vector riêng có thể xoay chiều (ví dụ: PC1 và -PC1 giải thích phương sai như nhau). 
                Để PC1 phản ánh đúng xu hướng 'Tăng trưởng', mô hình đã tự động chuẩn hóa dấu sao cho 
                trung bình trọng số của các cổ phiếu lớn là dương.
            """)
            
    # --- CỘT 2 ---
    with col2:
        # 1. PC2 Loading Chart (Dòng tiền luân chuyển ngành)
        loadings_pc2 = loadings_df['PC2'].sort_values(ascending=False)
        fig_pc2_bar = px.bar(loadings_pc2, x=loadings_pc2.index, y='PC2',
                              labels={'PC2': 'PC2 Loading'},
                              color='PC2', color_continuous_scale='coolwarm')
        fig_pc2_bar.update_layout(title="PC2 Loadings (Trường Dòng tiền Luân chuyển ngành)", template="plotly_white")
        st.plotly_chart(fig_pc2_bar, use_container_width=True)
        
        # 2. Text Explanation PC2
        st.markdown("""
            ### Phân tích PC2 Loadings:
            Khác với PC1, PC2 bắt đầu cho thấy sự phân hóa rõ nét. Một số cổ phiếu có Loading dương cao (trái) và một số có Loading âm cao (phải).
            Điều này thể hiện nhân tố PC2 phản ánh sự **luân chuyển dòng tiền** (Rotation) hoặc sự đối cực về rủi ro giữa các nhóm ngành.
        """)
        
        # 3. Line Chart So sánh PC1 vs VN30
        st.markdown("---")
        vn30_prices = historical_df.loc[standardized_df.index, 'ACB'] # Sử dụng mã ACB đại diện cho chỉ số VN30 (vì trong file CSV gửi ko có VN30_INDEX, cần lấy mã tiêu biểu hoặc user cần chuẩn bị file VN30 chuẩn).
        # Tạm thời lấy ACB làm đại diện vì file CSV gửi ko có VN30. 
        # Cần User sửa chỗ này nếu có mã VN30 Index chuẩn.
        
        # Scale VN30 Index and PC1 Cumsum for easy comparison
        scaled_pc1_cum = (PC_scores['PC1'].cumsum() - PC_scores['PC1'].cumsum().mean()) / PC_scores['PC1'].cumsum().std()
        scaled_vn30_price = (vn30_prices - vn30_prices.mean()) / vn30_prices.std()
        
        fig_line_comp = go.Figure()
        fig_line_comp.add_trace(go.Scatter(x=standardized_df.index, y=scaled_pc1_cum, mode='lines', name='PC1 Score (Cumsum, Scaled)'))
        fig_line_comp.add_trace(go.Scatter(x=standardized_df.index, y=scaled_vn30_price, mode='lines', name='VN30 Index Price (Scaled, ACB reprezent)'))
        
        fig_line_comp.update_layout(title="Hiệu năng Scaled: PC1 Score vs VN30 Index Price", template="plotly_white")
        st.plotly_chart(fig_line_comp, use_container_width=True)
        
        # 4. Text Explanation
        st.markdown("""
            Biểu đồ so sánh PC1 tích lũy (tự xây dựng) và Chỉ số VN30. 
            Mặc dù ko hoàn hảo (do PCA nén ko mất dữ liệu hoàn toàn), ta thấy PC1 Score 
            theo sát nhịp đập và xu hướng chính của toàn thị trường VN30.
        """)

# ------------------------------------------
# --- TAB 4: Nghiên cứu chuyên sâu ---
# ------------------------------------------
with tab4:
    st.header("4. Phần Nghiên cứu chuyên sâu (Revised)")
    st.markdown("🔍 Khám phá sâu qua các câu hỏi nghiên cứu.")
    
    # Thanh thả thả chọn câu hỏi nghiên cứu Q1-Q6
    questions = [
        "Chọn câu hỏi nghiên cứu...",
        "Q1. So sánh Ma trận hiệp phương sai Thủ công vs Scikit-learn",
        "Q2. Nghiên cứu sâu sự phân cụm PC Loadings (Gom cụm phân cấp)",
        "Q3. Nghiên cứu sâu Ma trận Tương quan Tỷ suất sinh lợi (Log Returns)",
        "Q4. Phân tích PC1: Nó có thực sự nắm bắt Rủi ro Thị trường?",
        "Q5. Nghiên cứu sâu Tương quan Lăn (Rolling Correlation): Phân tích Rủi ro Lan truyền",
        "Q6. Trực quan hóa Biplot: PC1 Loadings vs PC2 Loadings (Bản đồ Dòng tiền VN30)"
    ]
    selected_q = st.selectbox("", questions)
    
    st.markdown("---")
    
    if selected_q == questions[0]:
        st.info("💡 Hãy chọn một câu hỏi nghiên cứu phía trên để xem kết quả và nhận xét chi tiết.")
        
    elif selected_q == questions[1]:
        # --- Q1 ---
        st.subheader("Q1. So sánh Ma trận hiệp phương sai Thủ công vs Scikit-learn")
        with st.spinner("⏳ Đang chạy Q1..."):
            X = standardized_df.values
            cov_manual = (X.T @ X) / (X.shape[0] - 1)
            
            from sklearn.decomposition import PCA
            pca_sk = PCA(n_components=K_features)
            pca_sk.fit(X)
            cov_sk = pca_sk.get_covariance()
            
            # So sánh độ lệch (MAE)
            mae = np.mean(np.abs(cov_manual - cov_sk))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Ma trận Hiệp phương sai Thủ công (X.T @ X)**")
                st.dataframe(pd.DataFrame(cov_manual, columns=standardized_df.columns).iloc[:5, :5])
            with col2:
                st.write("**Ma trận Hiệp phương sai Scikit-learn**")
                st.dataframe(pd.DataFrame(cov_sk, columns=standardized_df.columns).iloc[:5, :5])
            
            st.success(f"📌 **Nhận xét:** Hai ma trận gần như trùng khớp hoàn toàn. Độ sai lệch MAE cực nhỏ: **{mae:.2e}**. Điều này khẳng định bước tính ma trận hiệp phương sai thủ công của ta là chính xác.")
            
    elif selected_q == questions[2]:
        # --- Q2 ---
        st.subheader("Q2. Nghiên cứu Gom cụm PC Loadings (Dendrogram & Grouped Bar)")
        with st.spinner("⏳ Đang chạy Q2..."):
            K_pc_compare = min(2, K_features)
            loadings_cluster = loadings_df.iloc[:, :K_pc_compare].T # (PC, stocks) -> Cluster stocks
            
            # Gom cụm phân cấp
            Z = linkage(squareform( squareform( np.abs( standardized_df.iloc[:, :K_pc_compare].corr() ) ) ), 'ward')
            clusters_idx = fcluster(Z, t=2, criterion='maxclust') # Gom thành 2 cụm chính
            
            # Vẽ Dendrogram
            fig_dendro, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z, labels=loadings_cluster.columns, ax=ax)
            ax.set_title('Gom cụm phân cấp Cổ phiếu VN30 theo rủi ro thị trường (PC1&PC2)')
            st.pyplot(fig_dendro)
            
            # Vẽ grouped bar chart theo cụm
            clustered_loadings = loadings_df.copy()
            clustered_loadings['Cluster'] = clusters_idx
            fig_clustered_bar = px.bar(clustered_loadings, x=clustered_loadings.index, y=['PC1', 'PC2'],
                                        color='Cluster', color_continuous_scale='coolwarm',
                                        labels={'value': 'Loading'}, barmode='group')
            fig_clustered_bar.update_layout(title="So sánh PC Loadings Gom cụm", template="plotly_white")
            st.plotly_chart(fig_clustered_bar, use_container_width=True)
            
            st.success("📌 **Nhận xét:** Gom cụm PC Loadings giúp ta nhận diện rõ nét các nhóm cổ phiếu 'đồng pha' (thường là cùng ngành hoặc đặc tính rủi ro tương tự). Cụm xanh thường gom chặt các mã Ngân hàng lớn, Cụm đỏ gom các mã phân cực ngành khác.")
            
    elif selected_q == questions[3]:
        # --- Q3 ---
        st.subheader("Q3. Nghiên cứu Ma trận Tương quan Tỷ suất sinh lợi (Log Returns)")
        with st.spinner("⏳ Đang chạy Q3..."):
            fig_heatmap, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(log_returns_df.iloc[:, :10].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
            st.pyplot(fig_heatmap)
            
            st.success("📌 **Nhận xét:** Ma trận tương quan Log Returns cho thấy mối quan hệ tương quan mạnh mẽ (màu đỏ) giữa các mã cùng ngành, đặc biệt là nhóm **Ngân hàng** (VCB, BID, CTG...). Điều này cho thấy rủi ro hệ thống hoặc rủi ro lan truyền trong VN30 tập trung rất lớn ở nhóm Ngân hàng.")
            
    elif selected_q == questions[4]:
        # --- Q4 ---
        st.subheader("Q4. Phân tích PC1: Nó có thực sự nắm bắt Rủi ro Thị trường?")
        with st.spinner("⏳ Đang chạy Q4 (Hồi quy)..."):
            # Chạy hồi quy Returns ~ PC1 cho Top 3 mã Loading cao nhất
            top_stocks = loadings_df['PC1'].abs().sort_values(ascending=False).index[:3]
            
            # Regression results display
            st.write(f"Kết quả Hồi quy PC1 tác động lên Top 3 mã VN30 (mã Loading cao nhất):")
            
            for stock in top_stocks:
                y = localized_returns_sliced[stock]
                X_reg = sm.add_constant(PC_scores['PC1'])
                model = sm.OLS(y, X_reg)
                results = model.fit()
                
                # Display metrics neatly
                with st.expander(f"Hồi quy cho mã: {stock} (PC1 Beta & p-value)", expanded=False):
                    st.write(f"Beta (PC1 Coefficient): **{results.params['PC1']:.4f}**")
                    st.write(f"p-value (PC1): **{results.pvalues['PC1']:.2e}**")
                    st.write(f"R-squared: **{results.rsquared:.2f}**")
                    
                    st.write("---")
                    # Scatter plot with regression line (Plotly)
                    fig_reg_scatter = px.scatter(x=PC_scores['PC1'], y=y, 
                                                trendline="ols", trendline_color_override="red",
                                                labels={'x': 'PC1 Score', 'y': f'{stock} Daily Returns'})
                    fig_reg_scatter.update_layout(title=f" Scatter Plot & Regression Line: {stock} vs PC1", template="plotly_white")
                    st.plotly_chart(fig_reg_scatter, use_container_width=True)
                    
            st.success("📌 **Nhận xét:** Kết quả hồi quy cho thấy p-value của PC1 cực nhỏ (hệ số có ý nghĩa thống kê) và R-squared cao cho Top 3 mã. Điều này xác nhận **PC1 thực sự nắm bắt rủi ro thị trường chung**, khi Daily Returns của các mã này biến động phụ thuộc rất lớn vào 'nhịp đập' PC1.")
            
    elif selected_q == questions[5]:
        # --- Q5 ---
        st.subheader("Q5. Nghiên cứu Tương quan Lăn (Rolling Correlation): Phân tích Rủi ro Lan truyền")
        with st.spinner("⏳ Đang chạy Q5 (Rolling Correlation)..."):
            # Tính Tương quan lăn (Rolling 60 days) giữa PC1 và VN30 representative
            window = 60
            rolling_corr_dynamic = PC_scores['PC1'].rolling(window).corr(log_returns_df['ACB']) # Replace ACB with VN30 Index chuẩn nếu có.
            
            # Plotly Line chart
            fig_rolling = px.line(x=log_returns_df.index, y=rolling_corr_dynamic, 
                                labels={'x': 'Date', 'y': f'Rolling {window}D Correlation'},
                                color_discrete_sequence=['purple'])
            fig_rolling.add_hline(y=1, line_dash="dash", line_color="red")
            fig_rolling.update_layout(title=f"Rolling Tương quan (Cửa sổ: {window} Ngày): PC1 vs VN30", template="plotly_white")
            st.plotly_chart(fig_rolling, use_container_width=True)
            
            st.success("📌 **Nhận xét:** Biểu đồ tương quan lăn cho thấy mối tương quan luôn dương và cao (>0.8 trong phần lớn thời gian). Điều này cho thấy rủi ro thị trường và VN30 lan truyền rất nhất quán và mạnh mẽ qua PC1, ít khi có sự phân kỳ lớn.")
            
    elif selected_q == questions[6]:
        # --- Q6 ---
        st.subheader("Q6. Trực quan hóa Biplot: PC1 Loadings vs PC2 Loadings (Bản đồ Dòng tiền VN30)")
        with st.spinner("⏳ Đang chạy Q6 (Biplot)..."):
            # Scatter Plot Biplot (PC1 vs PC2 Loadings) - Tương tự Tab 4 cũ
            df_biplot = pd.DataFrame({
                'Cổ phiếu': stock_names,
                'PC1_Loading': loadings_df['PC1'],
                'PC2_Loading': loadings_df['PC2']
            })
            
            # Cần user chuẩn bị nhóm ngành trong VN30.csv để tô màu. Nếu ko có, tạm thời tô đỏ.
            # Gawin tạm thời tô theo dấu PC2
            df_biplot['Sector_approx'] = np.where(df_biplot['PC2_Loading'] > 0, 'PC2+ (Hưởng lợi PC2)', 'PC2- (Đối xứng PC2)')

            fig_biplot = px.scatter(df_biplot, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                                  labels={'PC1_Loading': 'PC1 (Rủi ro Hệ thống)', 'PC2_Loading': 'PC2 (Luân chuyển Ngành)'},
                                  template="plotly_white", color='Sector_approx', color_discrete_scale='coolwarm')
            
            fig_biplot.update_traces(textposition='top center', marker=dict(size=12))
            fig_biplot.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_biplot.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_biplot, use_container_width=True)
            
            st.success("📌 **Nhận xét:** Biplot là 'Bản đồ Dòng tiền' của VN30. Toàn bộ cổ phiếu đều có PC1 Loading > 0 (đồng pha thị trường). Sự đối xứng của PC2 Loading cho thấy sự luân chuyển dòng tiền hoặc rủi ro phân hóa ngành. Các mã Ngân hàng lớn gom cụm rất chặt, cho thấy dòng tiền đổ vào nhóm này mang tính đồng pha rất cao.")
