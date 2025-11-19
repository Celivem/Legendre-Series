import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import eval_legendre
import pandas as pd
import io

st.set_page_config(page_title="勒讓德級數 (極速版)", layout="wide")

# --- CSS 優化滑桿體驗 ---
st.markdown("""
<style>
    .stSlider {padding-top: 20px;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ 勒讓德級數 (Legendre Series) - 極速渲染版")
st.markdown("此版本採用 **預先計算 (Pre-calculation)** 技術，拖動滑桿時僅進行矩陣切片，實現絲滑般的即時繪圖。")

# --- 1. 側邊欄設定 ---
st.sidebar.header("1. 訊號設定")
example_options = {
    "方波 (Square)": "where(x > 0, 1, 0)",
    "三角波 (Triangle)": "where(x > 0, x, 0)",
    "多週期方波": "sign(sin(4 * pi * x))",
    "連續三角波": "arcsin(sin(5 * x))",
    "高斯波包": "sin(15 * x) * exp(-5 * x**2)",
    "自訂": ""
}
choice = st.sidebar.radio("選擇範例：", list(example_options.keys()))
default_val = example_options[choice] if choice != "自訂" else "x"

func_str = st.sidebar.text_input("f(x) 表達式：", value=default_val)
max_N = st.sidebar.number_input("最大計算階數 (Max N)", value=50, min_value=10, max_value=200)

st.sidebar.markdown("---")
st.sidebar.info("設定好後，請按下方按鈕進行一次性計算。")

# --- 2. 核心計算引擎 (只在按鈕按下時執行) ---
def precompute_everything(func_expr, max_n_val, num_points=500):
    """
    一次性計算所有需要的數據：
    1. 目標函數值 (Target)
    2. 所有係數 (Coefficients 0 to Max)
    3. 所有多項式矩陣 (Polynomial Basis Matrix)
    """
    # A. 準備座標
    x_vals = np.linspace(-1, 1, num_points)
    theta_vals = np.linspace(0, 2 * np.pi, num_points)
    x_polar = np.cos(theta_vals)

    # B. 解析函數
    def f(x_in):
        allowed = {
            "x": x_in, "np": np, "sin": np.sin, "cos": np.cos, 
            "exp": np.exp, "pi": np.pi, "abs": np.abs, "sign": np.sign,
            "where": np.where, "arcsin": np.arcsin, "legendre": eval_legendre
        }
        return eval(func_expr, {"__builtins__": None}, allowed)

    # C. 計算目標值 (真值)
    try:
        y_target = f(x_vals)
        r_target = f(x_polar)
    except Exception as e:
        return None, f"函數解析錯誤: {e}"

    # D. 計算係數 (耗時步驟)
    coeffs = []
    data_list = []
    try:
        for n in range(max_n_val + 1):
            factor = (2 * n + 1) / 2
            integrand = lambda x: f(x) * eval_legendre(n, x)
            val, _ = quad(integrand, -1, 1, limit=50) # limit設小一點加速
            coeffs.append(factor * val)
            data_list.append({"n": n, "cn": factor * val})
    except Exception as e:
        return None, f"積分錯誤: {e}"

    # E. 預先計算多項式矩陣 (核心優化步驟!)
    # 形狀: (Max_N+1, num_points)
    # 這樣滑桿移動時不需要再呼叫 eval_legendre，只要查表即可
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
        "poly_matrix_x": poly_matrix_x,         # Cache Cartesian basis
        "poly_matrix_polar": poly_matrix_polar, # Cache Polar basis
        "df": pd.DataFrame(data_list)
    }
    return result, None

# --- 3. 互動邏輯 ---
if st.sidebar.button("🚀 執行計算 (Pre-compute)", type="primary"):
    with st.spinner(f"正在計算前 {max_N} 階係數與矩陣，請稍候..."):
        res, err = precompute_everything(func_str, max_N)
        if err:
            st.error(err)
        else:
            st.session_state['viz_data'] = res
            st.session_state['func_name'] = func_str
            st.rerun() # 強制刷新以顯示滑桿

# --- 4. 繪圖渲染層 (極輕量化) ---
if 'viz_data' in st.session_state:
    data = st.session_state['viz_data']
    
    # 確認當前的 max_N 是否與計算時一致 (避免改了側邊欄沒按計算)
    current_max_computed = len(data['coeffs']) - 1
    
    st.success(f"✅ 計算完成！目標函數：`{st.session_state.get('func_name', '')}` (已緩存 {current_max_computed} 階數據)")

    # --- 滑桿 (Slider) ---
    # 這裡的動作非常快，因為不做任何積分或函數生成
    n_select = st.slider("調整顯示階數 (N)", 0, current_max_computed, 5)

    # --- 極速計算 (Matrix Dot Product) ---
    # 數學原理： y = [c0, c1, ... cn] dot [P0(x), P1(x), ... Pn(x)]
    # 只需要切片，不需要重算
    
    coeffs_slice = data['coeffs'][:n_select+1]
    
    # 直角座標近似
    # (n+1) dot (n+1, 500) -> (500,)
    y_approx = np.dot(coeffs_slice, data['poly_matrix_x'][:n_select+1])
    
    # 極座標近似
    r_approx = np.dot(coeffs_slice, data['poly_matrix_polar'][:n_select+1])

    # --- 繪圖 ---
    fig = plt.figure(figsize=(14, 6))
    
    # 左圖
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(data['x_vals'], data['y_target'], 'k--', alpha=0.3, label='Target')
    ax1.plot(data['x_vals'], y_approx, 'r-', lw=2, label=f'Approx N={n_select}')
    ax1.set_title("Cartesian View")
    ax1.set_ylim(np.min(data['y_target'])-0.5, np.max(data['y_target'])+0.5)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右圖
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    ax2.plot(data['theta_vals'], np.abs(data['r_target']), 'k--', alpha=0.3)
    ax2.plot(data['theta_vals'], np.abs(r_approx), 'b-', lw=2)
    ax2.fill(data['theta_vals'], np.abs(r_approx), 'blue', alpha=0.1)
    ax2.set_title("Polar View (Directional)")
    ax2.set_rticks([])

    st.pyplot(fig)
    
    # --- 數據下載區 ---
    with st.expander("查看係數數據"):
        st.dataframe(data['df'].head(n_select+1).style.format({"cn": "{:.6f}"}))

else:
    st.info("👈 請在左側設定參數並按下「執行計算」")
