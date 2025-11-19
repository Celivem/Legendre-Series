import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import eval_legendre
import pandas as pd
import io

# --- 1. 頁面設定 ---
st.set_page_config(page_title="勒讓德級數視覺化 (高效能版)", layout="wide")

# CSS 微調：讓滑桿上方留點空間，比較好看
st.markdown("""
<style>
    .stSlider {padding-top: 20px;}
    h1 {margin-bottom: 0px;}
</style>
""", unsafe_allow_html=True)

st.title("🌊 勒讓德級數 (Legendre Series) 互動實驗室")
st.markdown(r"""
輸入函數 $f(x)$，系統將一次性計算所有係數。拖動滑桿可即時觀察不同階數的疊加結果。
$$f(x) \approx \sum_{n=0}^{N} c_n P_n(x)$$
""")

# --- 2. 側邊欄：範例選擇 (維持原位) ---
st.sidebar.header("⚡ 快速範例選擇")

example_options = {
    "自訂輸入": "",
    "--- 基礎波形 ---": "where(x > 0, 1, 0)", 
    "方波 (Step)": "where(x > 0, 1, 0)",
    "三角波 (Ramp)": "where(x > 0, x, 0)",
    "絕對值 (V-Shape)": "abs(x)",
    
    "--- 週期/震盪 ---": "sign(sin(4 * pi * x))", 
    "多週期方波": "sign(sin(4 * pi * x))",
    "連續三角波": "arcsin(sin(5 * x))",
    "高頻餘弦": "cos(5 * pi * x)",
    
    "--- 物理/調變 ---": "sin(15 * x) * exp(-5 * x**2)",
    "波包 (Wave Packet)": "sin(15 * x) * exp(-5 * x**2)",
    "AM 調變訊號": "(1 + 0.5 * cos(10 * x)) * cos(50 * x)",
    
    "--- 多極子 ---": "x",
    "偶極子 (Dipole)": "x",
    "四極子 (Quadrupole)": "3*x**2 - 1"
}

# 過濾掉分隔線選項
selectable_options = [k for k in example_options.keys() if not k.startswith("---")]
selected_label = st.sidebar.radio("選擇波形模版：", selectable_options)

# 根據選擇設定預設值
default_func = "where(x > 0, 1, 0)"
if selected_label != "自訂輸入":
    default_func = example_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.info("💡 **小提示**：極座標圖中的 $x$ 對應於 $\cos(\\theta)$。這在物理場型分析中非常常見。")

# --- 3. 主介面輸入區 (維持原位) ---
col_input, col_param = st.columns([3, 1])

with col_input:
    func_str = st.text_input("輸入 f(x) (支援 numpy 語法)", value=default_func)
with col_param:
    # 為了效能體驗，我們限制最大 N 不超過 200 (通常 50 就很夠了)
    max_N_input = st.number_input("最大計算階數 Max N", value=50, min_value=5, max_value=200, step=5)

# --- 4. 核心計算引擎 (預先計算並緩存) ---
def precompute_data(func_expr, max_n_val, num_points=500):
    """
    一次性執行積分與矩陣生成
    """
    # A. 準備座標
    x_vals = np.linspace(-1, 1, num_points)
    theta_vals = np.linspace(0, 2 * np.pi, num_points)
    x_polar = np.cos(theta_vals)

    # B. 解析函數
    def f(x_in):
        allowed = {
            "x": x_in, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, "sign": np.sign,
            "where": np.where, "heaviside": np.heaviside,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            "legendre": eval_legendre
        }
        return eval(func_expr, {"__builtins__": None}, allowed)

    # C. 計算目標值 (真值)
    try:
        y_target = f(x_vals)
        r_target = f(x_polar)
    except Exception as e:
        return None, f"函數解析錯誤: {e}"

    # D. 積分計算係數
    coeffs = []
    data_list = []
    try:
        for n in range(max_n_val + 1):
            factor = (2 * n + 1) / 2
            integrand = lambda x: f(x) * eval_legendre(n, x)
            # limit 稍微調低以加速大量計算
            val, _ = quad(integrand, -1, 1, limit=50)
            coeffs.append(factor * val)
            data_list.append({"Order (n)": n, "Coefficient (cn)": factor * val})
    except Exception as e:
        return None, f"積分過程錯誤: {e}"

    # E. 預先生成多項式矩陣 (加速關鍵)
    # 形狀: (Max_N+1, num_points)
    poly_matrix_x = np.zeros((max_n_val + 1, num_points))
    poly_matrix_polar = np.zeros((max_n_val + 1, num_points))
    
    for n in range(max_n_val + 1):
        poly_matrix_x[n, :] = eval_legendre(n, x_vals)
        poly_matrix_polar[n, :] = eval_legendre(n, x_polar)

    # 包裝結果
    result = {
        "x_vals": x_vals,
        "theta_vals": theta_vals,
        "y_target": y_target,
        "r_target": r_target,
        "coeffs": np.array(coeffs),
        "poly_matrix_x": poly_matrix_x,
        "poly_matrix_polar": poly_matrix_polar,
        "df": pd.DataFrame(data_list)
    }
    return result, None

# --- 5. 執行按鈕與狀態管理 ---
if st.button("🚀 執行運算 (Pre-compute)", type="primary"):
    with st.spinner(f"正在預先計算前 {max_N_input} 階的所有數據..."):
        res, err = precompute_data(func_str, max_N_input)
        
        if err:
            st.error(err)
            st.session_state['viz_data'] = None
        else:
            st.session_state['viz_data'] = res
            st.session_state['current_func'] = func_str

# --- 6. 視覺化呈現 (只要 session_state 有資料就顯示) ---
if st.session_state.get('viz_data'):
    data = st.session_state['viz_data']
    
    st.success(f"✅ 計算完成！現在可以拖動下方滑桿，享受即時渲染的效果。")
    st.markdown("---")

    # === 互動滑桿區 (瞬間反應) ===
    max_n_available = len(data['coeffs']) - 1
    
    # 滑桿直接改變 n_select，Streamlit 重新執行時只跑下面的繪圖，不跑積分
    n_select = st.slider("調整疊加階數 (N)", 0, max_n_available, max_n_available)
    
    # === 極速運算 (矩陣切片 + 點積) ===
    # 數學: y = [c0...cn] • [P0...Pn]
    c_slice = data['coeffs'][:n_select+1]
    
    y_approx = np.dot(c_slice, data['poly_matrix_x'][:n_select+1])
    r_approx = np.dot(c_slice, data['poly_matrix_polar'][:n_select+1])

    # === 繪圖 ===
    fig = plt.figure(figsize=(14, 6))
    
    # 左圖：直角座標
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(data['x_vals'], data['y_target'], 'k--', alpha=0.3, label='Target f(x)')
    ax1.plot(data['x_vals'], y_approx, 'r-', linewidth=2, label=f'Approx (N={n_select})')
    ax1.set_title("Cartesian View")
    ax1.set_xlabel("x")
    ax1.set_ylim(np.min(data['y_target'])-0.5, np.max(data['y_target'])+0.5)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # 右圖：極座標
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.plot(data['theta_vals'], np.abs(data['r_target']), 'k--', alpha=0.3, label='Target')
    ax2.plot(data['theta_vals'], np.abs(r_approx), 'b-', linewidth=2, label='Approx')
    ax2.fill(data['theta_vals'], np.abs(r_approx), 'blue', alpha=0.1)
    ax2.set_title("Polar View (Directional)")
    ax2.set_rticks([]) # 隱藏雜亂刻度
    
    st.pyplot(fig)

    # === 下載區 ===
    col_dl1, col_dl2 = st.columns(2)

    # 圖片下載
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    col_dl1.download_button("📥 下載當前圖表 (PNG)", img_buf, f"legendre_N{n_select}.png", "image/png")

    # CSV 下載
    df = data['df']
    csv_data = df.to_csv(index=False).encode('utf-8')
    col_dl2.download_button("📥 下載係數表 (CSV)", csv_data, "coefficients.csv", "text/csv")

    with st.expander("查看詳細係數"):
        # Highlighting current N row could be complex, just show data
        st.dataframe(df.style.format({"Coefficient (cn)": "{:.6f}"}))

elif not st.session_state.get('viz_data'):
    st.info("👈 請確認上方參數後，按下「執行運算」按鈕。")
