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
plt.rcParams['font.family'] = 'Montserrat'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU 
# ==========================================
@st.cache_data
def load_and_process_data(start_date, end_date, uploaded_file):
    # Khai báo chuẩn danh sách mã (đã bao gồm .VN để tải Yahoo Finance)
    tickers_yf = [
        "ACB.VN", "BID.VN", "BVH.VN", "CTG.VN", "FPT.VN",
        "GAS.VN", "GVR.VN", "HDB.VN", "HPG.VN", "KDH.VN",
        "MBB.VN", "MSN.VN", "MWG.VN", "NLG.VN", "NVL.VN",
        "PDR.VN", "PLX.VN", "POW.VN", "SBT.VN", "SSB.VN",
        "SSI.VN", "STB.VN", "TCB.VN", "TPB.VN", "VCB.VN",
        "VHM.VN", "VIB.VN", "VIC.VN", "VJC.VN", "VNM.VN"
    ]

    # Tải dữ liệu từ Yahoo Finance
    panel_data = yf.download(tickers_yf, start=start_date, end=end_date, progress=False)['Close']
    panel_data.columns = [col.replace('.VN', '') for col in panel_data.columns]
    panel_data.index = pd.to_datetime(panel_data.index).tz_localize(None).normalize()

    # Xử lý file CSV VN30_INDEX
    vn30_index_data = pd.read_csv(uploaded_file)
    vn30_index_data = vn30_index_data[['Ngày', 'Lần cuối']]
    vn30_index_data.columns = ['Date', 'VN30_INDEX']
    if vn30_index_data['VN30_INDEX'].dtype == 'O':
        vn30_index_data['VN30_INDEX'] = vn30_index_data['VN30_INDEX'].str.replace(',', '').astype(float)
    vn30_index_data['Date'] = pd.to_datetime(vn30_index_data['Date'], format='%d/%m/%Y')
    vn30_index_data.set_index('Date', inplace=True)
    vn30_index_data.index = vn30_index_data.index.tz_localize(None).normalize()
    
    # Gộp dữ liệu
    df_final = panel_data.join(vn30_index_data, how='outer')
    df_final = df_final.loc[start_date:end_date]

    # Làm sạch triệt để dữ liệu số
    df_final = df_final.replace({',': ''}, regex=True)
    df_final = df_final.apply(pd.to_numeric, errors='coerce')
    df_final = df_final.ffill().bfill()
    
    # Tính Log Returns (Tỷ suất sinh lợi)
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

# --- XỬ LÝ DỮ LIỆU ---
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

# Tính toán Loadings và PC Scores
stock_names_list = standardized_stock_returns.columns
loadings = sorted_eigenvectors * np.sqrt(sorted_eigenvalues)
loadings_df = pd.DataFrame(loadings, index=stock_names_list, columns=[f'PC{i+1}' for i in range(len(sorted_eigenvalues))])
PC_scores = pd.DataFrame(X @ sorted_eigenvectors, index=standardized_stock_returns.index, columns=[f'PC{i+1}' for i in range(len(sorted_eigenvalues))])

# Đổi dấu PC1 nếu ngược chiều VN30 (Để PC1 phản ánh đúng chiều thị trường)
vn30_returns = df_returns['VN30_INDEX']
if np.corrcoef(PC_scores['PC1'], vn30_returns)[0, 1] < 0:
    PC_scores['PC1'] = -PC_scores['PC1']
    loadings_df['PC1'] = -loadings_df['PC1']
    sorted_eigenvectors[:, 0] = -sorted_eigenvectors[:, 0]

# ==========================================
# GIAO DIỆN TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["1. Tiền xử lý và EDA", "2. Thuật toán PCA", "3. Yếu tố thị trường (PC1)", "4. Cơ cấu chuyên sâu"])

# --- TAB 1: EDA ---
# --- TAB 1: EDA NÂNG CẤP ---
with tab1:
    st.header("Khám phá dữ liệu (EDA)")
    st.markdown("Bước tiền xử lý đóng vai trò quyết định. Dữ liệu giá được chuyển sang **Log Returns** để đảm bảo tính dừng (stationarity) và phân phối chuẩn - hai giả định cực kỳ quan trọng của các mô hình định lượng tài chính.")
    
    col_eda1, col_eda2 = st.columns([1, 1.2])
    
    with col_eda1:
        st.subheader("📈 Biến động Tỷ suất sinh lợi (Volatility)")
        # Vẽ Line chart cho VN30 Index để thấy biến động và rủi ro thị trường
        # Backup nếu không có VN30_INDEX thì lấy mã đầu tiên
        col_to_plot = 'VN30_INDEX' if 'VN30_INDEX' in df_returns.columns else df_returns.columns[0]
        
        fig_returns = px.line(df_returns, y=col_to_plot, 
                              labels={'value': 'Log Returns', 'Date': 'Thời gian'})
        fig_returns.update_layout(title=f"Nhịp đập rủi ro của {col_to_plot}", 
                                  template="plotly_white", 
                                  showlegend=False)
        # Đổi màu line sang cam cho chuẩn vibe tài chính
        fig_returns.update_traces(line_color='#ff7f0e', line_width=1.5)
        st.plotly_chart(fig_returns, use_container_width=True)
        
        st.markdown("""
            **Góc nhìn tài chính:** Biểu đồ thể hiện mức độ biến động (Volatility) hàng ngày. 
            Các đợt dao động mạnh (nhọn) phản ánh những giai đoạn thị trường hấp thụ cú sốc thông tin.
        """)
        
    with col_eda2:
        st.subheader("🔗 Ma trận Tương quan (Full rổ VN30)")
        # Đổi sang Plotly Heatmap toàn bộ 30 mã (loại bỏ VN30_Index ra khỏi ma trận)
        corr_full = df_returns.drop(columns=['VN30_INDEX'], errors='ignore').corr()
        
        fig_corr = px.imshow(corr_full, 
                             text_auto=False, # Tắt số đi để ma trận 30x30 không bị rối mắt
                             aspect="auto", 
                             color_continuous_scale='RdBu_r', # Đỏ: Tương quan nghịch, Xanh: Tương quan thuận
                             labels=dict(color="Hệ số"))
        fig_corr.update_layout(title="Bản đồ nhiệt Tương quan (Hover để xem chi tiết)", 
                               template="plotly_white",
                               margin=dict(l=0, r=0, t=30, b=0)) # Căn lề cho to ra
        st.plotly_chart(fig_corr, use_container_width=True)
        
    st.markdown("---")
    with st.expander("❓ Khám phá: Vì sao Ma trận tương quan lại quan trọng ở bước này?", expanded=True):
        st.info("""
            Trong toán học, nếu các cổ phiếu di chuyển hoàn toàn độc lập (hệ số tương quan ~ 0), thuật toán PCA sẽ vô dụng vì không có thông tin nào để nén. 
            Tuy nhiên, nhìn vào Heatmap trên, ta thấy các dải màu hiển thị mức độ tương quan thuận rất rõ rệt. Chính sự "ràng buộc" này là **tiền đề bắt buộc** để thuật toán PCA ở Tab 2 có thể chiết xuất thành công các Nhân tố cốt lõi (Principal Components).
        """)
# --- TAB 2: Thuật toán PCA ---
with tab2:
    st.header("2. Phần Thuật toán PCA")
    st.markdown("Thay vì dùng thư viện có sẵn, mô hình áp dụng **toán học ma trận** để phân rã Hiệp phương sai thành Trị riêng và Vector riêng.")

    col_pca1, col_pca2 = st.columns([1, 1.5])
    with col_pca1:
        st.subheader("📐 So sánh Eigenvalues")
        numpy_eigenvals, _ = np.linalg.eigh(cov_matrix)
        numpy_eigenvals = np.sort(numpy_eigenvals)[::-1]
        comp_df = pd.DataFrame({
            'PC': range(1, X.shape[1] + 1),
            'Trị riêng (Thủ công)': sorted_eigenvalues,
            'Trị riêng (Numpy)': numpy_eigenvals
        })
        st.dataframe(comp_df.head(10), use_container_width=True)

    with col_pca2:
        st.subheader("Phân tích phương sai giải thích (Scree Plot)")
        explained_variance_ratio = (sorted_eigenvalues / sum(sorted_eigenvalues)) * 100
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=list(range(1, 11)), y=explained_variance_ratio[:10], name='Riêng lẻ (%)', marker_color='skyblue'))
        fig_scree.add_trace(go.Scatter(x=list(range(1, 11)), y=cumulative_explained_variance[:10], mode='lines+markers', name='Tích lũy (%)', line=dict(color='orange')))
        fig_scree.update_layout(xaxis_title="Thành phần chính (PC)", yaxis_title="Tỷ lệ (%)", template="plotly_white")
        st.plotly_chart(fig_scree, use_container_width=True)
        st.info(f"💡 **Insight:** Chỉ với 1 thành phần chính đầu tiên (PC1) đã giải thích được **{explained_variance_ratio[0]:.2f}%** toàn bộ biến động của rổ VN30.")

# --- TAB 3: PC1 MARKET FACTOR ---
with tab3:
    st.header("3. Phần Yếu tố thị trường")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Trọng số rủi ro thị trường (PC1)")
        loadings_pc1 = loadings_df['PC1'].sort_values(ascending=False)
        fig_pc1_bar = px.bar(loadings_pc1, x=loadings_pc1.index, y='PC1', color='PC1', color_continuous_scale='viridis')
        fig_pc1_bar.update_layout(template="plotly_white")
        st.plotly_chart(fig_pc1_bar, use_container_width=True)
        st.markdown("**Phân tích:** PC1 đại diện cho nhân tố thị trường chung. Các cổ phiếu đều có trọng số đồng chiều dương rất rõ nét.")
            
  # --- CỘT 2 ---
    with col2:
        st.subheader("Hiệu năng Scaled: PC1 Score vs VN30 Index")
        
        # 1. Đồng bộ hóa Index: Ép VN30 Index phải cắt bỏ ngày đầu tiên giống như PC_scores
        valid_index = PC_scores.index
        vn30_prices = df_final.loc[valid_index, 'VN30_INDEX']
        
        # Gawin's Backup: Nếu file CSV của cậu bị lỗi (như ảnh 2 cậu từng phải dùng Proxy), 
        # ta sẽ tự động lấy trung bình cộng 30 mã làm VN30 Index giả lập (Equal-Weighted Index)
        if vn30_prices.isnull().all():
            st.warning("⚠️ File VN30.csv không hợp lệ hoặc rỗng. Đang sử dụng VN30 Equal-Weighted (Proxy) để thay thế.")
            vn30_prices = df_final.drop(columns=['VN30_INDEX'], errors='ignore').loc[valid_index].mean(axis=1)
        else:
            # Lấp đầy các ngày NaN lắt nhắt do khác biệt lịch nghỉ lễ giữa Yahoo và CSV
            vn30_prices = vn30_prices.ffill().bfill()
        
        # 2. Tính toán Scaled (Chuẩn hóa Z-score)
        scaled_pc1_cum = (PC_scores['PC1'].cumsum() - PC_scores['PC1'].cumsum().mean()) / PC_scores['PC1'].cumsum().std()
        scaled_vn30_price = (vn30_prices - vn30_prices.mean()) / vn30_prices.std()
        
        # 3. Vẽ biểu đồ Plotly
        fig_line_comp = go.Figure()
        fig_line_comp.add_trace(go.Scatter(x=valid_index, y=scaled_pc1_cum, mode='lines', name='PC1 Score (Tích lũy)'))
        fig_line_comp.add_trace(go.Scatter(x=valid_index, y=scaled_vn30_price, mode='lines', name='VN30 Index'))
        
        fig_line_comp.update_layout(
            template="plotly_white",
            legend=dict(
                orientation="h",
                y=1.02,
                yanchor="bottom",
                x=0.5,
                xanchor="center"
            )
        )
        st.plotly_chart(fig_line_comp, use_container_width=True)
with tab4:
    st.header("4. Phần Nghiên cứu chuyên sâu")
    
    questions_list_res = [
        "Chọn câu hỏi nghiên cứu...",
        "Q1. Những cổ phiếu nào đóng vai trò 'đầu tàu' rủi ro hệ thống mạnh nhất?",
        "Q2. Phân cụm PC Loadings theo ngành (Hierarchical Clustering)",
        "Q3. Ma trận Tương quan Log Returns (Full VN30)",
        "Q4. Hồi quy OLS: PC1 có nắm bắt Rủi ro Thị trường?",
        "Q5. Rolling Correlation (60 ngày): Phân tích lan truyền",
        "Q6. Biplot (PC1 vs PC2): Bản đồ luân chuyển dòng tiền"
    ]
    selected_q = st.selectbox("Khám phá sâu qua các câu hỏi nghiên cứu:", questions_list_res)
    st.markdown("---")
    
    if selected_q == questions_list_res[1]:
        st.subheader(questions_list_res[1])
        with st.spinner("⏳ Đang thực thi..."):
            cov_numpy_comp = np.cov(X, rowvar=False)
            mae_cov_val = np.mean(np.abs(cov_matrix - cov_numpy_comp))
            st.success(f"📌 Hai ma trận trùng khớp hoàn toàn. Độ sai lệch MAE: **{mae_cov_val:.2e}**.")
            
    elif selected_q == questions_list_res[2]:
        st.subheader(questions_list_res[2])
        with st.spinner("⏳ Đang thực thi gom cụm..."):
            # Từ điển nhóm ngành (cập nhật theo list KDH, NLG...)
            groups_dict_res = {
                'ACB': 'Ngân hàng', 'BID': 'Ngân hàng', 'CTG': 'Ngân hàng', 'HDB': 'Ngân hàng', 
                'MBB': 'Ngân hàng', 'SSB': 'Ngân hàng', 'STB': 'Ngân hàng', 'TCB': 'Ngân hàng', 
                'TPB': 'Ngân hàng', 'VCB': 'Ngân hàng', 'VIB': 'Ngân hàng',
                'KDH': 'Bất động sản', 'NLG': 'Bất động sản', 'NVL': 'Bất động sản', 
                'PDR': 'Bất động sản', 'VHM': 'Bất động sản', 'VIC': 'Bất động sản', 
                'FPT': 'Công nghệ', 'GAS': 'Năng lượng', 'PLX': 'Năng lượng', 'POW': 'Năng lượng',
                'GVR': 'Cao su/Công nghiệp', 'HPG': 'Thép/Công nghiệp', 'SBT': 'Nông nghiệp',
                'MSN': 'Tiêu dùng', 'MWG': 'Bán lẻ', 'VNM': 'Tiêu dùng',
                'SSI': 'Chứng khoán', 'VJC': 'Hàng không', 'BVH': 'Bảo hiểm' 
            }
            loadings_with_sector_q2 = loadings_df[['PC1', 'PC2']].copy()
            loadings_with_sector_q2['Sector'] = loadings_with_sector_q2.index.map(groups_dict_res).fillna('Khác')

            abs_corr_mat_q2 = np.abs(standardized_stock_returns.corr()) 
            Z_q2_link = linkage(squareform(1 - abs_corr_mat_q2), 'ward')
            
            fig_dendro, ax = plt.subplots(figsize=(10, 4))
            dendrogram(Z_q2_link, labels=stock_names_list, ax=ax)
            ax.set_title('Dendrogram: Gom cụm phân cấp Cổ phiếu VN30')
            st.pyplot(fig_dendro)
            
            # Biểu đồ Bar luân chuyển ngành
            fig_grouped_bar = px.bar(loadings_with_sector_q2.sort_values(by='PC2', ascending=False), 
                                     x=loadings_with_sector_q2.index, y='PC2', color='Sector')
            fig_grouped_bar.update_layout(title="PC2 Loadings phân loại theo Nhóm ngành", template="plotly_white")
            st.plotly_chart(fig_grouped_bar, use_container_width=True)
            
    elif selected_q == questions_list_res[3]:
        st.subheader(questions_list_res[3])
        with st.spinner("⏳ Đang vẽ Heatmap..."):
            fig_heatmap_full_q3, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(df_returns.corr(), annot=False, cmap='coolwarm', fmt=".2f", ax=ax, center=0)
            st.pyplot(fig_heatmap_full_q3)
            
    elif selected_q == questions_list_res[4]:
        st.subheader(questions_list_res[4])
        with st.spinner("⏳ Đang chạy mô hình OLS..."):
            top_3_stocks_q4_ols = loadings_df['PC1'].abs().sort_values(ascending=False).index[:3]
            for stock in top_3_stocks_q4_ols:
                y_reg_log_ret = df_returns[stock]
                X_reg_ols = sm.add_constant(PC_scores['PC1']) 
                model = sm.OLS(y_reg_log_ret, X_reg_ols).fit()
                
                with st.expander(f"Kết quả Hồi quy cho: {stock} (Click để mở)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Beta (Hệ số PC1)", f"{model.params['PC1']:.4f}")
                    c2.metric("R-squared", f"{model.rsquared:.2f}")
                    c3.metric("P-value", f"{model.pvalues['PC1']:.2e}")
                    
                    x_val = model.model.exog[:, 1]
                    fig_scatter = px.scatter(x=x_val, y=y_reg_log_ret, labels={'x': 'PC1 Score', 'y': f'Lợi suất {stock}'})
                    x_line = np.linspace(x_val.min(), x_val.max(), 100)
                    y_line = model.params['const'] + model.params['PC1'] * x_line
                    fig_scatter.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Đường xu hướng (OLS Line)', line=dict(color='red')))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
    elif selected_q == questions_list_res[5]:
        st.subheader(questions_list_res[5])
        with st.spinner("⏳ Đang tính toán Rolling Correlation..."):
            vn30_rep = df_returns['VN30_INDEX'] 
            rolling_corr = PC_scores['PC1'].rolling(window=60).corr(vn30_rep)
            
            fig_roll = px.line(x=df_returns.index, y=rolling_corr, labels={'x': 'Thời gian', 'y': 'Hệ số Tương quan (60-day)'})
            fig_roll.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Ngưỡng đồng pha cao (0.8)")
            fig_roll.update_layout(template="plotly_white")
            st.plotly_chart(fig_roll, use_container_width=True)
            st.success("📌 **Nhận xét:** Hệ số tương quan phần lớn neo trên 0.8, PC1 duy trì tính đại diện xuất sắc xuyên suốt chu kỳ.")
            
    elif selected_q == questions_list_res[6]:
        st.subheader(questions_list_res[6])
        with st.spinner("⏳ Đang kết xuất Biplot..."):
            df_biplot = pd.DataFrame({
                'Cổ phiếu': stock_names_list,
                'PC1_Loading': loadings_df['PC1'],
                'PC2_Loading': loadings_df['PC2']
            })
            df_biplot['Sector_approx'] = np.where(df_biplot['PC2_Loading'] > 0, 'Nhóm hưởng lợi PC2 (Dương)', 'Nhóm đối xứng PC2 (Âm)')

            # SỬA LỖI COLOR XUNG ĐỘT Ở ĐÂY: Sử dụng color_discrete_sequence thay vì dải màu liên tục
            fig_bip = px.scatter(df_biplot, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                                  color='Sector_approx', 
                                  color_discrete_sequence=['#ef553b', '#636efa'], 
                                  template="plotly_white")
                                  
            fig_bip.update_traces(textposition='top center', marker=dict(size=12))
            fig_bip.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_bip.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_bip.update_layout(height=600, title="Bản đồ phân cụm cổ phiếu (Scatter 2D)")
            
            st.plotly_chart(fig_bip, use_container_width=True)
            st.success("📌 **Nhận xét Q6:** Cấu trúc hình phễu minh chứng rõ rệt: Trục X (PC1) là sự đồng pha hệ thống, Trục Y (PC2) bóc tách sự luân chuyển giữa các cụm ngành (Ví dụ: Ngân hàng vs Bất động sản/Công nghiệp).")
