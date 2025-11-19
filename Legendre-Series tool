import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import eval_legendre
import pandas as pd
import io

# --- 1. 頁面設定 ---
st.set_page_config(page_title="勒讓德級數視覺化", layout="wide")

# CSS 微調，讓數學公式顯示更清楚
st.markdown("""
<style>
    .stSlider {padding-top: 20px;}
</style>
""", unsafe_allow_html=True)

st.title("🌊 勒讓德級數 (Legendre Series) 互動實驗室")
st.markdown(r"""
此工具將計算函數 $f(x)$ 在區間 $[-1, 1]$ 上的展開：
$$
f(x) \approx \sum_{n=0}^{N} c_n P_n(x), \quad c_n = \frac{2n+1}{2} \int_{-1}^{1} f(x) P_n(x) dx
$$
包含 **直角座標** (波形擬合) 與 **極座標** (方向性場型) 雙視圖。
""")

# --- 2. 側邊欄：豐富的範例庫 ---
st.sidebar.header("⚡ 快速範例選擇")

# 這裡整合了之前討論的各種週期與特殊函數
example_options = {
    "自訂輸入": "",
    "--- 基礎波形 ---": "where(x > 0, 1, 0)", # Placeholder
    "方波 (Step)": "where(x > 0, 1, 0)",
    "三角波 (Ramp)": "where(x > 0, x, 0)",
    "絕對值 (V-Shape)": "abs(x)",
    
    "--- 週期/震盪函數 ---": "cos(5 * pi * x)", # Placeholder
    "多週期方波 (Square Train)": "sign(sin(4 * pi * x))",
    "連續三角波 (Triangle Wave)": "arcsin(sin(5 * x))",
    "高頻餘弦 (High Freq)": "cos(5 * pi * x)",
    
    "--- 物理/調變波形 ---": "sin(15 * x) * exp(-5 * x**2)", # Placeholder
    "波包 (Wave Packet)": "sin(15 * x) * exp(-5 * x**2)",
    "全波整流 (Rectified)": "abs(sin(3 * pi * x))",
    "AM 調變訊號": "(1 + 0.5 * cos(10 * x)) * cos(50 * x)",
    
    "--- 多極子模型 ---": "x", # Placeholder
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
st.sidebar.info("💡 **小提示**：極座標圖中的 $x$ 對應於 $\cos(\\theta)$。這在天線場型或原子軌域物理中非常常見。")

# --- 3. 主介面輸入 ---
col_input, col_param = st.columns([3, 1])

with col_input:
    func_str = st.text_input("輸入 f(x) (支援 numpy 語法)", value=default_func)
with col_param:
    max_N_input = st.number_input("最大計算階數 N", value=20, min_value=1, max_value=100, step=1)

# --- 4. 核心邏輯 (快取運算) ---
@st.cache_data(show_spinner=False)
def calculate_coefficients(func_expression, max_n):
    """
    計算勒讓德係數並回傳。
    使用 st.cache_data 避免滑動滑桿時重複積分。
    """
    # 定義安全的 eval 環境
    def f(x_val):
        allowed_locals = {
            "x": x_val, "np": np,
            # 基礎數學
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            # 邏輯與分段
            "where": np.where, "heaviside": np.heaviside,
            "maximum": np.maximum, "minimum": np.minimum,
            # 反三角 (用於生成三角波)
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            # 特殊
            "legendre": eval_legendre 
        }
        return eval(func_expression, {"__builtins__": None}, allowed_locals)

    coeffs = []
    data_table = []
    
    # 測試函數有效性
    try:
        _ = f(0.5)
    except Exception as e:
        return None, None, f"語法解析錯誤: {str(e)}"

    # 開始積分
    try:
        for n in range(max_n + 1):
            # 權重函數: (2n+1)/2
            factor = (2 * n + 1) / 2
            integrand = lambda x: f(x) * eval_legendre(n, x)
            
            # quad積分
            val, _ = quad(integrand, -1, 1, limit=100)
            cn = factor * val
            
            coeffs.append(cn)
            data_table.append({"Order (n)": n, "Coefficient (cn)": cn})
            
        return coeffs, data_table, None
        
    except Exception as e:
        return None, None, f"積分過程錯誤 (可能函數不收斂): {str(e)}"

# 輔助：僅用於繪圖時產生真值 (不積分)
def get_target_values(func_expression, x_arr):
    allowed_locals = {
            "x": x_arr, "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            "where": np.where, "heaviside": np.heaviside,
            "maximum": np.maximum, "minimum": np.minimum,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            "legendre": eval_legendre 
    }
    try:
        return eval(func_expression, {"__builtins__": None}, allowed_locals)
    except:
        return np.zeros_like(x_arr)

# --- 5. 執行邏輯 ---
if st.button("🚀 執行運算", type="primary"):
    st.session_state['run_analysis'] = True

if st.session_state.get('run_analysis'):
    
    with st.spinner("正在進行數值積分與矩陣運算..."):
        coeffs, data_table, error = calculate_coefficients(func_str, max_N_input)

    if error:
        st.error(error)
    else:
        # --- 互動滑桿區 ---
        st.markdown("### 🎛️ 階數觀察器")
        
        # 滑桿：使用者調整 N
        current_n = st.slider("拖動滑桿以改變疊加階數 (n)：", 0, max_N_input, max_N_input)
        
        # 準備繪圖數據
        # 1. 座標點
        x_vals = np.linspace(-1, 1, 500)
        theta_vals = np.linspace(0, 2 * np.pi, 500)
        x_polar = np.cos(theta_vals) # 將極座標角度轉回 x 變數

        # 2. 目標函數 (真值)
        y_target = get_target_values(func_str, x_vals)
        r_target = get_target_values(func_str, x_polar)

        # 3. 近似函數 (疊加)
        # 為了效能，我們只計算到當前選定的 current_n
        active_coeffs = coeffs[:current_n+1]
        
        # 利用廣播/向量化計算多項式值：形狀 [n+1, 500]
        # 注意：eval_legendre(n, x) 支援 x 為陣列
        # 我們可以用一個迴圈或列表推導，因為 n 通常不大 (<=100)
        poly_matrix_x = np.array([eval_legendre(n, x_vals) for n in range(current_n + 1)])
        poly_matrix_polar = np.array([eval_legendre(n, x_polar) for n in range(current_n + 1)])
        
        # 矩陣乘法求和: [1, n] dot [n, 500] -> [1, 500]
        y_approx = np.dot(active_coeffs, poly_matrix_x)
        r_approx = np.dot(active_coeffs, poly_matrix_polar)

        # --- 繪圖 ---
        plt.rcParams['axes.grid'] = True
        fig = plt.figure(figsize=(14, 6))
        
        # 左圖：直角座標
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x_vals, y_target, 'k--', alpha=0.3, linewidth=1.5, label='Target f(x)')
        ax1.plot(x_vals, y_approx, 'r-', linewidth=2, label=f'Approx (N={current_n})')
        ax1.set_title(f"Cartesian View\n$x$ vs $f(x)$", fontsize=14)
        ax1.set_xlabel("x")
        ax1.set_ylim(np.min(y_target)-0.5, np.max(y_target)+0.5)
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3)

        # 右圖：極座標
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        # 對於極座標半徑，通常取絕對值來表示強度，或保留正負值但用顏色區分
        # 這裡為了視覺一致性，顯示絕對值幅度，並填色
        ax2.plot(theta_vals, np.abs(r_target), 'k--', alpha=0.3, label='Target')
        ax2.plot(theta_vals, np.abs(r_approx), 'b-', linewidth=2, label='Approx')
        ax2.fill(theta_vals, np.abs(r_approx), 'blue', alpha=0.1)
        ax2.set_title(f"Polar View (Directional)\n$r = |\\sum c_n P_n(\\cos\\theta)|$", fontsize=14)
        ax2.set_rticks([]) # 隱藏徑向刻度使其更簡潔
        
        st.pyplot(fig)

        # --- 下載與數據區 ---
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)

        # 圖片下載
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        col_dl1.download_button("📥 下載圖表 (PNG)", img_buf, "legendre_viz.png", "image/png")

        # CSV 下載
        df_coeffs = pd.DataFrame(data_table)
        csv_data = df_coeffs.to_csv(index=False).encode('utf-8')
        col_dl2.download_button("📥 下載係數表 (CSV)", csv_data, "coefficients.csv", "text/csv")

        with st.expander("查看詳細係數數值"):
            st.dataframe(df_coeffs.style.format({"Coefficient (cn)": "{:.6f}"}))
