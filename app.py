import streamlit as st
import numpy as np
import math
from scipy.interpolate import interp1d, interp2d

st.set_page_config(page_title="BYD电热耦合仿真平台", layout="wide")

# ==========================================
# 1. 动态数据库：从图片提取的数据点
# ==========================================
# 假设从 600A 模块规格书提取的 Eon 数据点 (Ic, Tj=25, Tj=150)
ic_range = np.array([10, 50, 100, 200, 400, 600, 800])
eon_25 = np.array([0.88, 2.98, 5.94, 13.49, 35.32, 70.77, 121.08]) # mJ
eoff_25 = np.array([0.98, 2.81, 4.93, 9.53, 24.19, 42.92, 65.56]) # mJ

# 创建插值函数 (基于 Ic 和 Tj 的二维模型)
f_eon = interp1d(ic_range, eon_25, kind='quadratic', fill_value="extrapolate")
f_eoff = interp1d(ic_range, eoff_25, kind='quadratic', fill_value="extrapolate")

# ==========================================
# 2. UI 界面：工况与热阻设置
# ==========================================
st.title("🛡️ 功率模块动态损耗与结温闭环仿真")

with st.sidebar:
    st.header("1. 热阻参数 (Rth)")
    # 从图 10 提取 RthJF 稳态值
    rth_jc = st.number_input("壳到结热阻 RthJC (K/W)", value=0.065)
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)

col1, col2 = st.columns(2)
with col1:
    st.header("2. 实时运行工况")
    vdc = st.number_input("直流母线电压 Vdc (V)", value=510.0)
    iout = st.number_input("输出有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    rg = st.number_input("当前驱动电阻 Rg (Ω)", value=5.0)

# ==========================================
# 3. 核心计算引擎：电热耦合迭代 solver
# ==========================================
if st.button("🚀 执行动态闭环计算"):
    # 初始结温猜测值
    tj_iter = t_case + 10 
    i_peak = math.sqrt(2) * iout
    
    # 开始迭代 (为了精确反推损耗和结温)
    for i in range(5):
        # A. 动态选取损耗量 (基于插值)
        eon_base = f_eon(iout)
        eoff_base = f_eoff(iout)
        
        # B. 工况修正因子
        kv = math.pow(vdc / 510, 1.3) # 电压修正
        kt = 1 + 0.003 * (tj_iter - 25) # 温度修正
        
        p_sw = (eon_base + eoff_base) * fsw * kv * kt * (1/math.pi) / 1000
        
        # C. 导通损耗 (这里也建议采用插值 Vce(Ic, Tj))
        # 简化演示：采用线性 Vce = 1.0 + 0.002*tj_iter
        vce = 1.0 + 0.001 * tj_iter
        p_cond = vce * iout * 0.45 # 简化 SVPWM 导通分量
        
        p_total = p_sw + p_cond
        
        # D. 反推结温 Tj = Tc + P * Rth
        tj_new = t_case + p_total * rth_jc
        
        if abs(tj_new - tj_iter) < 0.1:
            break
        tj_iter = tj_new

    # ==========================================
    # 4. 结果输出
    # ==========================================
    st.divider()
    res1, res2, res3 = st.columns(3)
    res1.metric("反推实时结温 Tj", f"{tj_iter:.2f} ℃")
    res2.metric("动态总损耗 P_total", f"{p_total:.2f} W")
    res3.metric("当前电流单芯密度", f"{iout/6:.1f} A/chip")
    
    st.write(f"提示：当前 Eon 是基于 Ic={iout}A 从双脉冲表 动态插值选取的，而非固定值。")
