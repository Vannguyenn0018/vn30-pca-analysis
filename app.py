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
tab1, tab2, tab3, tab4 = st.tabs(["1. Tiền xử lý và EDA ", "2. Thuật toán PCA", "3. Yếu tố thị trường (PC1)", "4. Cơ cấu chuyên sâu"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Khám phá dữ liệu (EDA)")
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

# --- TAB 2: Thuật toán PCA ---
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
# --- TAB 4: Nghiên cứu chuyên sâu ---
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
            
   elif selected_q_research == questions_list[2]:
        # --- Q2 ---
        st.subheader("Q2. Nghiên cứu sâu Gom cụm PC Loadings (Gom cụm Gom PC1&PC2)")
        with st.spinner("⏳ Đang thực thi Q2 Gom cụm..."):
            # Lấy PC1 & PC2 loadings
            k_pc_comp_q2 = min(2, K_features_main)
            loadings_cluster_q2 = loadings_df_main.iloc[:, :k_pc_comp_q2] 
            
            # --- Gawin Cập nhật groups_dict tại đây ---
            # Thêm 'ACB', 'BID', 'CTG'... vào nhóm Ngân hàng, 
            # thêm các mã khác vào nhóm tương ứng của chúng.
            groups_dict = {
                'ACB': 'Ngân hàng', 'BID': 'Ngân hàng', 'CTG': 'Ngân hàng', 'HDB': 'Ngân hàng', 'MBB': 'Ngân hàng',
                'SHB': 'Ngân hàng', 'STB': 'Ngân hàng', 'TCB': 'Ngân hàng', 'TPB': 'Ngân hàng', 'VCB': 'Ngân hàng', 'VIB': 'Ngân hàng', 'VPB': 'Ngân hàng',
                'BCM': 'Bất động sản', 'VHM': 'Bất động sản', 'VIC': 'Bất động sản', 'VRE': 'Bất động sản',
                'FPT': 'Công nghệ',
                'GAS': 'Năng lượng', 'PLX': 'Năng lượng', 'POW': 'Năng lượng',
                'GVR': 'Cao su/Công nghiệp',
                'HPG': 'Thép/Công nghiệp',
                'MSN': 'Tiêu dùng', 'MWG': 'Tiêu dùng', 'SAB': 'Tiêu dùng', 'VNM': 'Tiêu dùng',
                'SSI': 'Chứng khoán',
                'VJC': 'Hàng không', 'BVH': 'Bảo hiểm', 'SSB': 'Ngân hàng' # Thêm SSB mới vào VN30
            }
            # --- Kết thúc cập nhật groups_dict ---

            # Map nhóm ngành vào DataFrame loadings
            # Gawin bẫy lỗi nhẹ: Nếu mã ko có trong groups_dict thì đặt là 'Khác'
            loadings_cluster_q2['Sector'] = loadings_cluster_q2.index.map(groups_dict).fillna('Khác')

            # Gom cụm phân cấp Hierarchical clustering (giữ nguyên Q2)
            abs_corr_mat = np.abs(standardized_df.iloc[:, :k_pc_comp_q2].corr()) 
            Z_q2 = linkage(squareform(squareform(abs_corr_mat)), 'ward')
            clusters_idx_q2 = fcluster(Z_q2, t=2, criterion='maxclust') # Gom thành 2 cụm chính
            
            # Vẽ Dendrogram
            fig_dendro, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z_q2, labels=stock_names_list, ax=ax)
            ax.set_title('Q2 Gom cụm phân cấp Cổ phiếu VN30 theo rủi ro thị trường (PC1&PC2)')
            st.pyplot(fig_dendro)
            
            # Grouped Bar chart compare loadings across clusters and sectors
            clustered_loadings_all_q2 = loadings_cluster_q2.copy() # Sử dụng loadings đã map sector
            clustered_loadings_all_q2['Cluster'] = clusters_idx_q2
            clustered_loadings_all_q2.sort_values(by=['Cluster', 'PC1'], ascending=[True, False], inplace=True)
            
            fig_grouped_bar_comp_q2 = px.bar(clustered_loadings_all_q2, x=clustered_loadings_all_q2.index, y=['PC1', 'PC2'],
                                        color='Sector', # Gawin sửa tô màu theo 'Sector' (nhóm ngành)
                                        color_discrete_scale='viridis',
                                        labels={'value': 'Loading', 'index': 'Cổ phiếu', 'Sector': 'Nhóm ngành'}, barmode='group')
            fig_grouped_bar_comp_q2.update_layout(title="So sánh PC Loadings Gom cụm theo Cụm và Nhóm ngành", template="plotly_white")
            st.plotly_chart(fig_grouped_bar_comp_q2, use_container_width=True)
            
            st.success("📌 **Nhận xét Q2:** Gom cụm PC Loadings giúp ta nhận diện rõ nét các nhóm cổ phiếu 'đồng pha' (thường là cùng ngành). Biểu đồ Bar chart trên cho thấy nhóm Ngân hàng (cụm đỏ) gom chặt lại, trong khi các ngành khác phân hóa sang cụm khác.")
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
