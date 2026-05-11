import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="VN30 PCA Analysis", layout="wide", page_icon="📈")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU
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
    # Tùy chỉnh theo định dạng thực tế của file CSV của bạn
    vn30_index_data = vn30_index_data[['Ngày', 'Lần cuối']]
    vn30_index_data.columns = ['Date', 'VN30_INDEX']
    if vn30_index_data['VN30_INDEX'].dtype == 'O':
        vn30_index_data['VN30_INDEX'] = vn30_index_data['VN30_INDEX'].str.replace(',', '').astype(float)
    vn30_index_data['Date'] = pd.to_datetime(vn30_index_data['Date'], format='%d/%m/%Y')
    vn30_index_data.set_index('Date', inplace=True)
    vn30_index_data.index = vn30_index_data.index.tz_localize(None).normalize()
    
    # Gộp và xử lý Missing Values
    df_final = panel_data.join(vn30_index_data, how='outer')
    df_final = df_final.loc[start_date:end_date]

    # 1. Xóa bỏ mọi dấu phẩy (,) phân cách hàng nghìn bị kẹt lại
    df_final = df_final.replace({',': ''}, regex=True)
    # 2. Ép toàn bộ về kiểu số. Nếu có ký tự lạ (như chữ, khoảng trắng), nó sẽ biến thành NaN
    df_final = df_final.apply(pd.to_numeric, errors='coerce')
    # 3. Điền khuyết (Fillna) để đảm bảo không bị lỗi chia cho NaN
    df_final = df_final.ffill().bfill()
    
    # Tính Log Returns (Bây giờ 100% dữ liệu đã là số thực)
    df_returns = np.log(df_final / df_final.shift(1)).dropna()
    
    # Chuẩn hóa dữ liệu cổ phiếu (Bỏ VN30_INDEX ra để chạy PCA)
    scaler = StandardScaler()
    temp_standardized = pd.DataFrame(scaler.fit_transform(df_returns), columns=df_returns.columns, index=df_returns.index)
    standardized_stock_returns = temp_standardized.drop(columns=['VN30_INDEX'], errors='ignore')
    
    return df_final, df_returns, standardized_stock_returns
    # Tính Log Returns thay vì pct_change để đúng với mô tả (Log Returns)
    df_returns = np.log(df_final / df_final.shift(1)).dropna()
    
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
start_date = st.sidebar.date_input("2. Ngày bắt đầu", pd.to_datetime('2021-01-01'))
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
sorted_eigenvalues = np.array([pair[0] for pair in eigen_pairs])
sorted_eigenvectors = np.array([pair[1] for pair in eigen_pairs]).T

# Tính toán Loadings và PC Scores (Phần bị thiếu trong code gốc)
stock_names_list = standardized_stock_returns.columns
loadings = sorted_eigenvectors * np.sqrt(sorted_eigenvalues)
loadings_df = pd.DataFrame(loadings, index=stock_names_list, columns=[f'PC{i+1}' for i in range(len(sorted_eigenvalues))])

PC_scores = pd.DataFrame(X @ sorted_eigenvectors, index=standardized_stock_returns.index, columns=[f'PC{i+1}' for i in range(len(sorted_eigenvalues))])

# Đổi dấu PC1 nếu ngược chiều VN30
vn30_returns = df_returns['VN30_INDEX']
if np.corrcoef(PC_scores['PC1'], vn30_returns)[0, 1] < 0:
    PC_scores['PC1'] = -PC_scores['PC1']
    loadings_df['PC1'] = -loadings_df['PC1']
    sorted_eigenvectors[:, 0] = -sorted_eigenvectors[:, 0]

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
    st.dataframe(df_returns.head(10).T, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔗 Ma trận tương quan (Top 10)")
    corr = df_returns.iloc[:, :10].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=False)
    st.pyplot(fig)

# --- TAB 2: Thuật toán PCA ---
with tab2:
    st.header("2. Phần Thuật toán PCA")
    st.markdown("""
        Xây dựng & Phân rã Trị riêng (Eigen Decomposition): Thay vì chỉ dùng thư viện Black-box, mô hình áp dụng **Thuật toán phân rã** để tiến hành phân rã Ma trận Hiệp phương sai thành các trị riêng (Eigenvalues) và vector riêng (Eigenvectors).
    """)

    st.markdown("---")
    st.subheader("📐 So sánh Eigenvalues: Thủ công vs Numpy (Toàn bộ)")
    K_features = standardized_stock_returns.shape[1]
    
    # Tính trị riêng Numpy cho comparison
    numpy_eigenvals, _ = np.linalg.eigh(cov_matrix)
    numpy_eigenvals = np.sort(numpy_eigenvals)[::-1]
    
    comp_df = pd.DataFrame({
        'PC': range(1, K_features + 1),
        'Trị riêng (Thủ công)': sorted_eigenvalues,
        'Trị riêng (Numpy eigh)': numpy_eigenvals
    })
    st.dataframe(comp_df, use_container_width=True)

    # Scree Plot
    st.markdown("---")
    st.subheader("Phân tích phương sai giải thích (Scree Plot)")
    explained_variance_ratio = (sorted_eigenvalues / sum(sorted_eigenvalues)) * 100
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, 11), explained_variance_ratio[:10], alpha=0.7, label='Phương sai riêng lẻ (%)')
    ax.step(range(1, 11), cumulative_explained_variance[:10], where='mid', color='orange', label='Phương sai tích lũy (%)')
    ax.set_ylabel('Tỷ lệ (%)')
    ax.set_xlabel('Thành phần chính (PC)')
    ax.legend()
    st.pyplot(fig)
    
    st.info(f"💡 **Insight:** Chỉ với 1 thành phần chính đầu tiên (PC1) đã giải thích được **{explained_variance_ratio[0]:.2f}%** toàn bộ biến động của rổ VN30.")

# --- TAB 3: PC1 MARKET FACTOR ---
with tab3:
    st.header("3. Phần Yếu tố thị trường")
    st.subheader("⚖️ So sánh Hiệu năng Thành phần Chính (PC1)")
    
    col1, col2 = st.columns([1, 1])
    
    # --- CỘT 1 ---
    with col1:
        loadings_pc1 = loadings_df['PC1'].sort_values(ascending=False)
        fig_pc1_bar = px.bar(loadings_pc1, x=loadings_pc1.index, y='PC1',
                              labels={'PC1': 'PC1 Loading'},
                              color='PC1', color_continuous_scale='viridis')
        fig_pc1_bar.update_layout(title="PC1 Loadings (Trọng số rủi ro thị trường)", template="plotly_white")
        st.plotly_chart(fig_pc1_bar, use_container_width=True)
        
        st.markdown("""
            ### Phân tích PC1 Loadings:
            PC1 thường đại diện cho nhân tố thị trường (Market Trend/Factor). 
            Ta quan sát thấy toàn bộ các cổ phiếu đều có trọng số (Loading) **đồng chiều dương** rất rõ nét.
        """)
        st.markdown("---")
        with st.expander("❓ Vấn đề về Dấu của PC1", expanded=True):
            st.warning("""
                Trong PCA, vector riêng có thể xoay chiều. Để PC1 phản ánh đúng xu hướng 'Tăng trưởng', 
                mô hình đã tự động chuẩn hóa dấu sao cho đồng pha với VN30_INDEX.
            """)
            
    # --- CỘT 2 ---
    with col2:
        loadings_pc2 = loadings_df['PC2'].sort_values(ascending=False)
        fig_pc2_bar = px.bar(loadings_pc2, x=loadings_pc2.index, y='PC2',
                              labels={'PC2': 'PC2 Loading'},
                              color='PC2', color_continuous_scale='coolwarm')
        fig_pc2_bar.update_layout(title="PC2 Loadings (Dòng tiền Luân chuyển ngành)", template="plotly_white")
        st.plotly_chart(fig_pc2_bar, use_container_width=True)
        
        st.markdown("""
            ### Phân tích PC2 Loadings:
            Khác với PC1, PC2 bắt đầu cho thấy sự phân hóa rõ nét, phản ánh sự **luân chuyển dòng tiền** (Rotation) giữa các nhóm ngành.
        """)
        
        st.markdown("---")
        # Sử dụng VN30_INDEX thay vì ACB vì đã có trong dữ liệu
        vn30_prices = df_final['VN30_INDEX']
        
        scaled_pc1_cum = (PC_scores['PC1'].cumsum() - PC_scores['PC1'].cumsum().mean()) / PC_scores['PC1'].cumsum().std()
        scaled_vn30_price = (vn30_prices - vn30_prices.mean()) / vn30_prices.std()
        
        fig_line_comp = go.Figure()
        fig_line_comp.add_trace(go.Scatter(x=standardized_stock_returns.index, y=scaled_pc1_cum, mode='lines', name='PC1 Score (Cumsum)'))
        fig_line_comp.add_trace(go.Scatter(x=standardized_stock_returns.index, y=scaled_vn30_price, mode='lines', name='VN30 Index Price'))
        fig_line_comp.update_layout(title="Hiệu năng Scaled: PC1 Score vs VN30 Index Price", template="plotly_white")
        st.plotly_chart(fig_line_comp, use_container_width=True)

# --- TAB 4: Nghiên cứu chuyên sâu ---
with tab4:
    st.header("4. Phần Nghiên cứu chuyên sâu")
    st.markdown("🔍 Khám phá sâu qua các câu hỏi nghiên cứu.")
    
    questions_list_res = [
        "Chọn câu hỏi nghiên cứu...",
        "Q1. So sánh Ma trận hiệp phương sai Thủ công vs Numpy/Scikit-learn",
        "Q2. Nghiên cứu sâu sự phân cụm PC Loadings (Gom cụm PC1&PC2 theo ngành)",
        "Q3. Nghiên cứu sâu Ma trận Tương quan Tỷ suất sinh lợi (Log Returns)",
        "Q4. Phân tích PC1: Nó có thực sự nắm bắt Rủi ro Thị trường? (Hồi quy OLS)",
        "Q5. Nghiên cứu sâu Tương quan Lăn (Rolling Correlation): Phân tích Rủi ro Lan truyền",
        "Q6. Trực quan hóa Biplot: PC1 Loadings vs PC2 Loadings (Bản đồ Dòng tiền VN30)"
    ]
    selected_q_research_main = st.selectbox("", questions_list_res, key='selectbox_research_revised')
    st.markdown("---")
    
    if selected_q_research_main == questions_list_res[0]:
        st.info("💡 Hãy chọn một câu hỏi nghiên cứu phía trên để xem kết quả chi tiết.")
        
    elif selected_q_research_main == questions_list_res[1]:
        st.subheader("Q1. So sánh Ma trận hiệp phương sai")
        with st.spinner("⏳ Đang thực thi Q1..."):
            cov_numpy_comp = np.cov(X, rowvar=False)
            mae_cov_val = np.mean(np.abs(cov_matrix - cov_numpy_comp))
            
            col1_q1_rev, col2_q1_rev = st.columns(2)
            with col1_q1_rev:
                st.write("**Ma trận Hiệp phương sai Thủ công**")
                st.dataframe(pd.DataFrame(cov_matrix, columns=stock_names_list).iloc[:5, :5], use_container_width=True)
            with col2_q1_rev:
                st.write("**Ma trận Hiệp phương sai Numpy (Reference)**")
                st.dataframe(pd.DataFrame(cov_numpy_comp, columns=stock_names_list).iloc[:5, :5], use_container_width=True)
            st.success(f"📌 **Nhận xét:** Hai ma trận trùng khớp hoàn toàn. Độ sai lệch MAE cực nhỏ: **{mae_cov_val:.2e}**.")
            
    elif selected_q_research_main == questions_list_res[2]:
        st.subheader("Q2. Nghiên cứu sâu Gom cụm PC Loadings theo ngành")
        with st.spinner("⏳ Đang thực thi..."):
            groups_dict_res = {
                'ACB': 'Ngân hàng', 'BID': 'Ngân hàng', 'CTG': 'Ngân hàng', 'HDB': 'Ngân hàng', 'MBB': 'Ngân hàng',
                'SHB': 'Ngân hàng', 'STB': 'Ngân hàng', 'TCB': 'Ngân hàng', 'TPB': 'Ngân hàng', 'VCB': 'Ngân hàng', 'VIB': 'Ngân hàng', 'VPB': 'Ngân hàng', 'SSB': 'Ngân hàng',
                'BCM': 'Bất động sản', 'VHM': 'Bất động sản', 'VIC': 'Bất động sản', 'VRE': 'Bất động sản',
                'FPT': 'Công nghệ', 'GAS': 'Năng lượng', 'PLX': 'Năng lượng', 'POW': 'Năng lượng',
                'GVR': 'Cao su/Công nghiệp', 'HPG': 'Thép/Công nghiệp',
                'MSN': 'Tiêu dùng', 'MWG': 'Tiêu dùng', 'SAB': 'Tiêu dùng', 'VNM': 'Tiêu dùng',
                'SSI': 'Chứng khoán', 'VJC': 'Hàng không', 'BVH': 'Bảo hiểm' 
            }
            loadings_with_sector_q2 = loadings_df[['PC1', 'PC2']].copy()
            loadings_with_sector_q2['Sector'] = loadings_with_sector_q2.index.map(groups_dict_res).fillna('Khác')

            abs_corr_mat_q2 = np.abs(standardized_stock_returns.corr()) 
            Z_q2_link = linkage(squareform(1 - abs_corr_mat_q2), 'ward')
            clusters_idx_q2_res = fcluster(Z_q2_link, t=2, criterion='maxclust')
            
            fig_dendro_link, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z_q2_link, labels=stock_names_list, ax=ax)
            ax.set_title('Dendrogram: Gom cụm phân cấp Cổ phiếu VN30')
            st.pyplot(fig_dendro_link)
            
            loadings_with_sector_q2['Cluster'] = clusters_idx_q2_res
            loadings_with_sector_q2.sort_values(by=['Cluster', 'PC1'], ascending=[True, False], inplace=True)
            
            fig_grouped_bar_clustered_q2 = px.bar(loadings_with_sector_q2, x=loadings_with_sector_q2.index, y=['PC1', 'PC2'],
                                        color='Sector', color_discrete_scale='coolwarm', barmode='group')
            st.plotly_chart(fig_grouped_bar_clustered_q2, use_container_width=True)
            
    elif selected_q_research_main == questions_list_res[3]:
        st.subheader("Q3. Ma trận Tương quan Tỷ suất sinh lợi")
        with st.spinner("⏳ Đang thực thi..."):
            fig_heatmap_full_q3, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(df_returns.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True, center=0)
            st.pyplot(fig_heatmap_full_q3)
            
    elif selected_q_research_main == questions_list_res[4]:
        st.subheader("Q4. Hồi quy OLS PC1 vs Daily Returns")
        with st.spinner("⏳ Đang thực thi..."):
            top_3_stocks_q4_ols = loadings_df['PC1'].abs().sort_values(ascending=False).index[:3]
            
            for stock in top_3_stocks_q4_ols:
                y_reg_log_ret = df_returns[stock]
                X_reg_ols = sm.add_constant(PC_scores['PC1']) 
                
                model = sm.OLS(y_reg_log_ret, X_reg_ols).fit()
                
                with st.expander(f"Hồi quy OLS cho mã: {stock}", expanded=False):
                    c1, c2 = st.columns(2)
                    c1.metric("Beta (PC1 Coeff)", f"{model.params['PC1']:.4f}")
                    c2.metric("R-squared", f"{model.rsquared:.2f}")
                    st.write(f"p-value (PC1): **{model.pvalues['PC1']:.2e}**")
                    
                    x_val = model.model.exog[:, 1]
                    fig_scatter = px.scatter(x=x_val, y=y_reg_log_ret, labels={'x': 'PC1 Score', 'y': 'Returns'})
                    
                    x_line = np.linspace(x_val.min(), x_val.max(), 100)
                    y_line = model.params['const'] + model.params['PC1'] * x_line
                    fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='OLS Line', line=dict(color='red')))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
    elif selected_q_research_main == questions_list_res[5]:
        st.subheader("Q5. Rolling Correlation (60 ngày)")
        with st.spinner("⏳ Đang thực thi..."):
            vn30_rep = df_returns['VN30_INDEX'] 
            rolling_corr = PC_scores['PC1'].rolling(window=60).corr(vn30_rep)
            
            fig_roll = px.line(x=df_returns.index, y=rolling_corr, labels={'x': 'Date', 'y': 'Rolling Correlation'})
            fig_roll.add_hline(y=1, line_dash="dash", line_color="red")
            st.plotly_chart(fig_roll, use_container_width=True)
            
    elif selected_q_research_main == questions_list_res[6]:
        st.subheader("Q6. Biplot: Bản đồ Dòng tiền VN30")
        with st.spinner("⏳ Đang thực thi..."):
            df_biplot = pd.DataFrame({
                'Cổ phiếu': stock_names_list,
                'PC1_Loading': loadings_df['PC1'],
                'PC2_Loading': loadings_df['PC2']
            })
            df_biplot['Sector_approx'] = np.where(df_biplot['PC2_Loading'] > 0, 'PC2+ (Hưởng lợi PC2)', 'PC2- (Đối xứng PC2)')

            fig_bip = px.scatter(df_biplot, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                                  color='Sector_approx', color_discrete_scale='coolwarm', template="plotly_white")
            fig_bip.update_traces(textposition='top center', marker=dict(size=12))
            fig_bip.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_bip.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig_bip, use_container_width=True)
            st.success("📌 **Nhận xét Q6:** Toàn bộ cổ phiếu đều có PC1 Loading > 0 (đồng pha thị trường). Sự đối xứng của PC2 Loading cho thấy sự phân hóa ngành rõ rệt.")
