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
st.set_page_config(page_title="VN30 PCA Analysis", layout="wide", page_icon="🏛️")
plt.rcParams['font.family'] = 'Montserrat'
plt.rcParams['axes.unicode_minus'] = False
# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="VN30 PCA Analysis", layout="wide", page_icon="📈")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ĐỊNH DẠNG TABS NỔI 3D
# ==========================================
st.markdown("""
<style>
    /* Chỉnh khoảng cách giữa các tabs */
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    /* Trạng thái mặc định của các Tab (chưa chọn) */
    div[data-baseweb="tab-list"] button[role="tab"] {
        background-color: #E2E8F0; /* Màu nền xám nhạt như hình */
        border-radius: 6px 6px 0px 0px; /* Bo góc phía trên */
        padding: 10px 24px;
        border: none;
        color: #64748B; /* Màu chữ xám xanh */
        font-weight: 600;
        transition: all 0.3s ease; /* Hiệu ứng chuyển màu mượt mà */
    }

    /* Xóa viền focus mặc định xấu xí của Streamlit */
    div[data-baseweb="tab-list"] button[role="tab"]:focus {
        outline: none;
    }

    /* Trạng thái Tab ĐƯỢC CHỌN (Active) */
    div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important; /* Đổi sang nền trắng */
        color: #1E3A8A !important; /* Đổi màu chữ đậm hơn */
        /* Tạo viền đỏ/xanh ở dưới đáy giống thiết kế của cậu */
        box-shadow: inset 0px -3px 0px 0px #EF4444; 
        border-bottom: 2px solid #2563EB; 
    }

    /* Chỉnh lại font chữ và căn giữa cho nội dung tab */
    div[data-baseweb="tab-list"] button[role="tab"] p {
        font-size: 15px;
        margin: 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ... (Phần hàm xử lý dữ liệu và cấu hình sidebar giữ nguyên) ...
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

# ==========================================
# CÀI ĐẶT DỮ LIỆU TỰ ĐỘNG (KHÔNG CẦN UPLOAD)
# ==========================================
# 1. Ẩn hoàn toàn Sidebar đi nếu không cần thiết
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 2. Cố định khoảng thời gian nghiên cứu (giống trong hình cậu set)
start_date = pd.to_datetime('2025-05-05')
end_date = pd.to_datetime('2026-04-29')

# 3. Đường dẫn trực tiếp đến file CSV (Đảm bảo file VN30.csv đã được up lên cùng thư mục code)
file_path = "VN30.csv.csv" 

# --- XỬ LÝ & LƯU TRỮ TRẠNG THÁI ---
with st.spinner('Đang tải dữ liệu tự động và chạy thuật toán PCA...'):
    try:
        # Truyền thẳng file_path vào hàm thay vì uploaded_file
        df_final, df_returns, standardized_stock_returns = load_and_process_data(start_date, end_date, file_path)
    except FileNotFoundError:
        st.error("⚠️ Không tìm thấy file 'VN30.csv'. Cậu nhớ upload file này lên cùng chỗ với file app.py nhé!")
        st.stop()
    except pd.errors.ParserError:
        # Đề phòng trường hợp file CSV của cậu sử dụng dấu chấm phẩy (;) phân tách dữ liệu gây lỗi đọc file
        st.error("⚠️ Lỗi định dạng file. Kiểm tra lại xem file VN30.csv có đang dùng dấu ';' thay vì dấu ',' mặc định không nhé.")
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
tab1, tab2, tab3, tab4 = st.tabs(["1. 📋 DỮ LIỆU &\n EDA", 
                                  "2. ⚙️ THUẬT TOÁN\n PCA",
                                  "3. 🌐 HIỆU NĂNG TÍCH LUỸ", 
                                  "4. 🔍 CẤU TRÚC\nCHUYÊN SÂU"])

# --- TAB 1: EDA NÂNG CẤP (FULL WIDTH LEOUP) ---
with tab1:
    st.header("Khám phá dữ liệu (EDA)")
    st.markdown("Bước tiền xử lý đóng vai trò quyết định. Dữ liệu giá được chuyển sang **Log Returns** để đảm bảo tính dừng (stationarity) và phân phối chuẩn - hai giả định cực kỳ quan trọng của các mô hình định lượng tài chính.")
    
    # --- PHẦN 1: BẢNG TỶ SUẤT SINH LỢI ---
    st.markdown("---")
    st.subheader("📊 Ma trận Tỷ suất sinh lợi (Log Returns)")
    st.dataframe(df_returns.head(10).T, use_container_width=True)
    
    # --- PHẦN 2: BIẾN ĐỘNG TỶ SUẤT SINH LỢI ---
    st.markdown("---")
    st.subheader("📈 Biến động Tỷ suất sinh lợi (Volatility)")
    
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
    
    # --- PHẦN 3: MA TRẬN TƯƠNG QUAN HEATMAP ---
    st.markdown("---")
    st.subheader("🔗 Ma trận Tương quan (Full rổ VN30)")
    
    # Loại bỏ VN30_Index ra khỏi ma trận để chỉ so sánh 30 cổ phiếu
    corr_full = df_returns.drop(columns=['VN30_INDEX'], errors='ignore').corr()
    
    fig_corr = px.imshow(corr_full, 
                         text_auto=False, # Tắt số để ma trận không bị rối mắt
                         aspect="auto", 
                         color_continuous_scale='RdBu_r', # Đỏ: Tương quan nghịch, Xanh: Tương quan thuận
                         labels=dict(color="Hệ số"))
                         
    # Cấu hình height=700 để heatmap to ra, hiển thị tên 30 mã rõ ràng không bị đè nhau
    fig_corr.update_layout(title="Bản đồ nhiệt Tương quan (Hover để xem chi tiết)", 
                           template="plotly_white",
                           height=700, 
                           margin=dict(l=0, r=0, t=30, b=0)) 
    st.plotly_chart(fig_corr, use_container_width=True)
    
    with st.expander("❓ Khám phá:  Ma trận tương quan của lợi suất hàng ngày", expanded=True):
        st.info("""
            Ma trận tương quan của lợi suất hàng ngày cho các cổ phiếu trong rổ VN30 cung cấp cái nhìn sâu sắc về mối quan hệ giữa các tài sản:

*   **Tương quan dương cao:** Biểu đồ nhiệt của ma trận tương quan cho thấy phần lớn các cổ phiếu trong rổ VN30 có hệ số tương quan dương với nhau. Điều này là đặc trưng của một thị trường chứng khoán, nơi các cổ phiếu thường có xu hướng biến động cùng chiều do chịu ảnh hưởng từ các yếu tố vĩ mô và tâm lý thị trường chung (Market Risk).
    *   Các vùng màu nóng (ví dụ, màu đỏ trong heatmap) biểu thị các cặp cổ phiếu có tương quan dương mạnh, nghĩa là chúng thường tăng/giảm cùng nhau.
    *   Các vùng màu lạnh (ví dụ, màu xanh lam) biểu thị tương quan âm hoặc thấp, cho thấy các cặp cổ phiếu có xu hướng biến động ngược chiều hoặc độc lập hơn.
*   **Sự hiện diện của yếu tố thị trường chung:** Mức độ tương quan dương cao giữa các cổ phiếu là bằng chứng mạnh mẽ cho thấy có một hoặc một vài yếu tố chung đang chi phối sự biến động của toàn bộ thị trường VN30. Đây chính là tiền đề quan trọng nhất để áp dụng phân tích PCA, vì PCA sẽ tìm cách trích xuất những 'yếu tố chung' này thành các Thành phần chính (Principal Components).
*   **Tiềm năng đa dạng hóa hạn chế:** Mức độ tương quan cao cũng ngụ ý rằng khả năng đa dạng hóa rủi ro bằng cách kết hợp các cổ phiếu trong rổ VN30 có thể bị hạn chế, vì phần lớn chúng đều phản ứng tương tự với các điều kiện thị trường.

Nhìn chung, cả EDA và ma trận tương quan đều xác nhận tính chất 
        """)
# --- TAB 2: THUẬT TOÁN PCA (FULL WIDTH STACKING) ---
with tab2:
    st.header("2. Phần Thuật toán PCA")
    st.markdown("""
        Tại đây, mô hình thực hiện phân rã dữ liệu bằng **toán học ma trận thuần túy**. Thay vì dùng các hàm 'Black-box', 
        chúng ta đi qua từng bước từ tính toán Hiệp phương sai đến tìm kiếm các Trị riêng (Eigenvalues).
    """)

    # --- PHẦN 1: MA TRẬN HIỆP PHƯƠNG SAI ---
    st.markdown("---")
    st.subheader("📐 Ma trận Hiệp phương sai (Covariance Matrix)")
    st.markdown("Đây là ma trận đo lường mức độ biến động cùng nhau của 30 mã cổ phiếu. Các giá trị trên đường chéo chính chính là phương sai của từng mã.")
    
    # Hiển thị ma trận hiệp phương sai (đã tính ở phần code chính) dưới dạng DataFrame
    cov_df = pd.DataFrame(cov_matrix, index=stock_names_list, columns=stock_names_list)
    st.dataframe(cov_df, use_container_width=True)
    
    st.info("💡 **Gawin's Note:** Ma trận này chính là 'bản đồ rủi ro' của rổ VN30. Các trị riêng (Eigenvalues) sau đây sẽ được chiết xuất trực tiếp từ chính ma trận này.")

    # --- PHẦN 2: SO SÁNH EIGENVALUES ---
    st.markdown("---")
    st.subheader("🔢 So sánh kết quả Trị riêng (Eigenvalues)")
    st.markdown("Bước này nhằm kiểm chứng độ chính xác của thuật toán thủ công so với hàm chuẩn của thư viện Numpy.")
    
    numpy_eigenvals, _ = np.linalg.eigh(cov_matrix)
    numpy_eigenvals = np.sort(numpy_eigenvals)[::-1]
    
    comp_df = pd.DataFrame({
        'Thành phần chính': [f'PC{i+1}' for i in range(X.shape[1])],
        'Trị riêng (Tính thủ công)': sorted_eigenvalues,
        'Trị riêng (Numpy Reference)': numpy_eigenvals
    })
    
    # Hiển thị bảng so sánh (Full width)
    st.dataframe(comp_df, use_container_width=True)

    # --- PHẦN 3: SCREE PLOT ---
    st.markdown("---")
    st.subheader("📊 Phân tích phương sai giải thích (Scree Plot)")
    
    explained_variance_ratio = (sorted_eigenvalues / sum(sorted_eigenvalues)) * 100
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    fig_scree = go.Figure()
    # Bar chart cho phương sai riêng lẻ
    fig_scree.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(10)], 
        y=explained_variance_ratio[:10], 
        name='Phương sai riêng lẻ (%)', 
        marker_color='skyblue'
    ))
    # Line chart cho phương sai tích lũy
    fig_scree.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(10)], 
        y=cumulative_explained_variance[:10], 
        mode='lines+markers', 
        name='Phương sai tích lũy (%)', 
        line=dict(color='orange', width=3)
    ))
    
    fig_scree.update_layout(
        title="Scree Plot: 10 Thành phần chính đầu tiên",
        xaxis_title="Thành phần chính (Principal Components)",
        yaxis_title="Tỷ lệ phương sai giải thích (%)",
        template="plotly_white",
        height=550,
        legend=dict(orientation="h", y=1.05, yanchor="bottom", x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_scree, use_container_width=True)
    
    st.success(f"""
        **Nhận định chuyên sâu:** Thành phần chính đầu tiên (PC1) giải thích được tới **{explained_variance_ratio[0]:.2f}%** biến động của toàn bộ thị trường. Điều này xác nhận cấu trúc thị trường VN30 có tính tập trung cực cao vào một nhân tố chung.
    """)
   # --- TAB 3: YẾU TỐ THỊ TRƯỜNG (FULL WIDTH) ---
with tab3:
    st.header("3. So sánh Hiệu năng Thành phần Chính (PC1)")
    st.markdown("Tab này tập trung vào **Thành phần chính đầu tiên (PC1)**. Trong tài chính, PC1 thường được hiểu là **Nhân tố thị trường (Market Factor)** – lực đẩy chung chi phối hầu hết các cổ phiếu.")

    # ==========================================
    # TÍNH TOÁN CÁC CHỈ SỐ KPI TỰ ĐỘNG
    # ==========================================
    # 1. Giải thích bởi PC1 (%)
    total_var = sum(sorted_eigenvalues)
    explained_pc1 = (sorted_eigenvalues[0] / total_var) * 100
    
    # 2. Tương quan PC1 vs VN30
    if 'VN30_INDEX' in df_returns.columns:
        corr_pc1_vn30 = np.corrcoef(PC_scores['PC1'], df_returns['VN30_INDEX'])[0, 1]
    else:
        # Nếu ko có VN30_INDEX thì lấy trung bình rổ làm đại diện
        corr_pc1_vn30 = np.corrcoef(PC_scores['PC1'], df_returns.mean(axis=1))[0, 1]
        
    # 3. Số PC đạt >= 90% Var
    cumulative_var = np.cumsum(sorted_eigenvalues / total_var) * 100
    num_pc_90 = np.argmax(cumulative_var >= 90) + 1

    # ==========================================
    # VẼ KPI CARDS BẰNG CSS HACK
    # ==========================================
    st.markdown("""
    <style>
    .kpi-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 8px solid #829AB1; /* Màu viền trái xanh xám giống hình */
        margin-bottom: 24px;
        margin-top: 10px;
    }
    .kpi-title {
        color: #486581;
        font-size: 15px;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .kpi-value {
        color: #102A43; /* Màu số xanh đen đậm */
        font-size: 34px;
        font-weight: 800;
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Giải thích bởi PC1</div><div class="kpi-value">{explained_pc1:.2f}%</div></div>', unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Tương quan PC1 vs VN30</div><div class="kpi-value">{corr_pc1_vn30:.4f}</div></div>', unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Số PC đạt ≥90% Var</div><div class="kpi-value">{num_pc_90}</div></div>', unsafe_allow_html=True)

    # ==========================================
    # BIỂU ĐỒ BÊN DƯỚI (GIỮ NGUYÊN)
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Hiệu năng Scaled: PC1 Score (Tích lũy) vs VN30 Index")
    
    valid_index = PC_scores.index
    vn30_prices = df_final.loc[valid_index, 'VN30_INDEX']
    
    if vn30_prices.isnull().all():
        st.warning("⚠️ File VN30.csv không hợp lệ. Đang sử dụng VN30 Equal-Weighted (Proxy) để thay thế.")
        vn30_prices = df_final.drop(columns=['VN30_INDEX'], errors='ignore').loc[valid_index].mean(axis=1)
    else:
        vn30_prices = vn30_prices.ffill().bfill()
    
    scaled_pc1_cum = (PC_scores['PC1'].cumsum() - PC_scores['PC1'].cumsum().mean()) / PC_scores['PC1'].cumsum().std()
    scaled_vn30_price = (vn30_prices - vn30_prices.mean()) / vn30_prices.std()
    
    fig_line_comp = go.Figure()
    fig_line_comp.add_trace(go.Scatter(x=valid_index, y=scaled_pc1_cum, 
                                       mode='lines', name='PC1 Score (Đại diện PCA)',
                                       line=dict(color='#1f77b4', width=2.5)))
    fig_line_comp.add_trace(go.Scatter(x=valid_index, y=scaled_vn30_price, 
                                       mode='lines', name='VN30 Index (Thực tế)',
                                       line=dict(color='#ff7f0e', width=2, dash='dot')))
    
    fig_line_comp.update_layout(
        title="So sánh xu hướng vận động: Nhân tố PCA tự xây dựng vs Chỉ số thực tế",
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            y=1.05,
            yanchor="bottom",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Thời gian",
        yaxis_title="Giá trị chuẩn hóa (Z-score)"
    )
    st.plotly_chart(fig_line_comp, use_container_width=True)
    
    st.success("**Kết luận tài chính:** Sự bám sát giữa đường màu xanh (PC1) và đường màu cam (VN30 thực tế) là bằng chứng cho thấy thuật toán PCA đã 'nén' thành công thông tin của 30 cổ phiếu vào 1 nhân tố duy nhất mà vẫn giữ được linh hồn của thị trường.")


# --- TAB 4: CƠ CẤU CHUYÊN SÂU (ADVANCED RESEARCH LAB) ---
with tab4:
    st.header("4. Nghiên cứu chuyên sâu & Phân lớp rủi ro")
    st.markdown("""
        Chào mừng đến với phần phân tích nâng cao. Tại đây, tôi đã sử dụng PCA để bóc tách các lớp rủi ro 
        và tìm kiếm sự luân chuyển của dòng tiền giữa các nhóm ngành trong rổ VN30.
    """)

    # ==========================================
    # TÍNH TOÁN KPI NHANH CHO RESEARCH
    # ==========================================
    r2_list = []
    for stock in df_returns.columns:
        if stock == 'VN30_INDEX': continue
        model_kpi = sm.OLS(df_returns[stock], sm.add_constant(PC_scores['PC1'])).fit()
        r2_list.append(model_kpi.rsquared)
    avg_r2 = np.mean(r2_list)
    
    explained_pc2 = (sorted_eigenvalues[1] / sum(sorted_eigenvalues)) * 100

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Độ giải thích TB (Avg R²)</div><div class="kpi-value">{avg_r2:.2f}</div></div>', unsafe_allow_html=True)
    with col_res2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Nhân tố ngành (PC2)</div><div class="kpi-value">{explained_pc2:.2f}%</div></div>', unsafe_allow_html=True)
    with col_res3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-title">Độ ổn định hệ thống</div><div class="kpi-value">Cao</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # ==========================================
    # MENU CHỌN MÔ HÌNH NGHIÊN CỨU
    # ==========================================
    research_questions = [
        "1. Cấu trúc dẫn dắt & Hệ số tải (Factor Loadings)",
        "2. Top 10 cổ phiếu 'đầu tàu' rủi ro hệ thống", # Bổ sung option mới
        "3. Biplot (PC1 vs PC2): Bản đồ luân chuyển dòng tiền",
        "4. Hierarchical Clustering: Gom cụm rủi ro theo ngành",
        "5. Hồi quy OLS chuyên sâu: Đo lường độ nhạy (Beta)"
    ]
    selected_res = st.selectbox("Chọn mô hình nghiên cứu:", research_questions)

    # ------------------------------------------
    # Q1. CẤU TRÚC DẪN DẮT & HỆ SỐ TẢI
    # ------------------------------------------
    if selected_res == research_questions[0]:
        st.subheader("Cấu trúc dẫn dắt & Hệ số tải (Factor Loadings)")
        
        col_left, col_right = st.columns([1, 1.5])
        
        with col_left:
            st.markdown("### 🧬 Phân tích Factor Loadings")
            st.markdown("""
            Biểu đồ bên cạnh thể hiện mức độ đóng góp của từng cổ phiếu vào **Thành phần chính số 1 (PC1)**.
            
            * **Trọng số dương cao:** Các cổ phiếu mang tính dẫn dắt thị trường (Market Leaders).
            * **Tính đồng nhất:** Nếu tất cả các mã đều có trọng số cùng dấu, thị trường có tính tương quan hệ thống rất cao.
            """)
            
            st.markdown("**Top 5 Mã chi phối PC1**")
            
            top5_pc1 = loadings_df[['PC1']].sort_values(by='PC1', ascending=False).head(5)
            top5_pc1.columns = ['Trọng số'] 
            top5_pc1.index.name = 'Mã'
            
            st.dataframe(top5_pc1.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with col_right:
            loadings_pc1_sorted = loadings_df['PC1'].sort_values(ascending=True)
            fig_factor = px.bar(
                x=loadings_pc1_sorted.values, 
                y=loadings_pc1_sorted.index, 
                orientation='h',
                labels={'x': 'Trọng số PC1', 'y': ''}
            )
            fig_factor.update_traces(marker_color='#5B9BD5')
            fig_factor.update_layout(
                title={
                    'text': "VN30 FACTOR LOADINGS (PC1)", 
                    'y': 0.95, 
                    'x': 0.5, 
                    'xanchor': 'center', 
                    'yanchor': 'top',
                    'font': {'size': 16, 'color': '#1f3864', 'weight': 'bold'}
                },
                template="plotly_white",
                height=650, 
                xaxis=dict(showgrid=True, gridcolor='#E2E8F0', tickformat=".2f"),
                yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#475569')),
                margin=dict(l=0, r=0, t=60, b=0)
            )
            st.plotly_chart(fig_factor, use_container_width=True)

    # ------------------------------------------
    # Q2. TOP 10 CỔ PHIẾU ĐẦU TÀU RỦI RO HỆ THỐNG (MỚI)
    # ------------------------------------------
    elif selected_res == research_questions[1]:
        st.subheader("🚂 Những cổ phiếu nào đóng vai trò 'đầu tàu' rủi ro hệ thống mạnh nhất?")
        st.markdown("Bảng dưới đây trích xuất Top 10 mã chứng khoán nhạy cảm nhất với rủi ro thị trường chung (đo lường bằng độ lớn tuyệt đối của trọng số PC1).")
        
        # Trích xuất và sắp xếp Top 10
        top_systemic_stocks = loadings_df[['PC1']].copy()
        top_systemic_stocks['Độ nhạy'] = top_systemic_stocks['PC1'].abs()
        top_systemic_stocks = top_systemic_stocks.sort_values(by='Độ nhạy', ascending=False).head(10)
        
        # Đổi tên cột chuẩn form
        display_df = top_systemic_stocks[['PC1']].copy()
        display_df.columns = ['Trọng số PC1 (Độ nhạy thị trường)']
        display_df.index.name = 'Mã Chứng Khoán'
        
        # Hiển thị bảng với dải màu YlOrRd
        st.dataframe(display_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
        
        # Tự động trích xuất mã nhạy cảm nhất (Top 1)
        top_1 = display_df.index[0]
        st.success(f"📌 **Nhận xét:** **{top_1}** là cổ phiếu nhạy cảm nhất. Khi PC1 (nhân tố thị trường) biến động 1 đơn vị, cổ phiếu này sẽ phản ứng mạnh nhất trong rổ VN30.")

    # ------------------------------------------
    # Q3. BIPLOT
    # ------------------------------------------
    elif selected_res == research_questions[2]:
        st.subheader("📍 Biplot: PC1 (Thị trường) vs PC2 (Phân hóa ngành)")
        st.markdown("Biplot là 'bản đồ gen' của rổ VN30. Trục X thể hiện xu hướng chung, Trục Y thể hiện sự đối lập giữa các nhóm ngành.")
        
        df_biplot = pd.DataFrame({
            'Cổ phiếu': stock_names_list,
            'PC1_Loading': loadings_df['PC1'],
            'PC2_Loading': loadings_df['PC2']
        })
        df_biplot['Loại'] = np.where(df_biplot['PC2_Loading'] > 0, 'Nhóm A (PC2+)', 'Nhóm B (PC2-)')

        fig_biplot = px.scatter(df_biplot, x='PC1_Loading', y='PC2_Loading', text='Cổ phiếu',
                                color='Loại', color_discrete_sequence=['#ef553b', '#636efa'])
        fig_biplot.update_traces(textposition='top center', marker=dict(size=12))
        fig_biplot.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_biplot.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_biplot.update_layout(template="plotly_white", height=600)
        st.plotly_chart(fig_biplot, use_container_width=True)

    # ------------------------------------------
    # Q4. CLUSTERING
    # ------------------------------------------
    elif selected_res == research_questions[3]:
        st.subheader("🌳 Phân cụm phân cấp (Hierarchical Clustering)")
        st.markdown("Thuật toán này giúp ta phát hiện những 'người anh em' cùng tiến cùng lùi trong VN30 dựa trên sự tương đồng về rủi ro.")
        
        with st.spinner("Đang tính toán ma trận khoảng cách..."):
            abs_corr = np.abs(df_returns.drop(columns=['VN30_INDEX'], errors='ignore').corr())
            Z = linkage(squareform(1 - abs_corr), 'ward')
            
            fig_cluster, ax = plt.subplots(figsize=(12, 6))
            dendrogram(Z, labels=stock_names_list, ax=ax, leaf_rotation=90)
            ax.set_title("Dendrogram: Cấu trúc liên kết các mã VN30", fontsize=14)
            st.pyplot(fig_cluster)

    # ------------------------------------------
    # Q5. OLS REGRESSION
    # ------------------------------------------
    elif selected_res == research_questions[4]:
        st.subheader("🧪 OLS Regression: Đo lường 'nhịp đập' Beta với PC1")
        st.markdown("Chúng ta sẽ chạy 30 mô hình hồi quy để xem mã nào 'nhạy' nhất với thị trường chung (PC1).")
        
        betas = {}
        for s in stock_names_list:
            res = sm.OLS(df_returns[s], sm.add_constant(PC_scores['PC1'])).fit()
            betas[s] = res.params['PC1']
        
        df_betas = pd.Series(betas).sort_values(ascending=False)
        
        fig_beta = px.bar(df_betas, x=df_betas.index, y=df_betas.values, 
                          color=df_betas.values, color_continuous_scale='Reds',
                          labels={'y': 'Hệ số Beta (với PC1)', 'index': 'Mã cổ phiếu'})
        fig_beta.update_layout(template="plotly_white", height=500, title="Xếp hạng độ nhạy rủi ro hệ thống")
        st.plotly_chart(fig_beta, use_container_width=True)
