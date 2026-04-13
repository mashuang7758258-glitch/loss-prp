import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD模块电热耦合对标平台", layout="wide")
st.title("🔬 功率模块动态损耗及结温闭环对标平台")

# --- 1. 原始测试数据矩阵录入 (支持实时修改) ---
st.header("1. 原始测试矩阵录入 (Double Pulse & Datasheet)")
col_data1, col_data2 = st.columns(2)

with col_data1:
    st.subheader("⚡ 双脉冲测试能量矩阵 (mJ)")
    # 提供可编辑表格
    dp_df = pd.DataFrame({
        'Ic (A)': [10, 50, 100, 200, 400, 600, 800],
        'Eon (mJ)': [0.88, 2.98, 5.94, 13.49, 35.32, 70.77, 121.08],
        'Eoff (mJ)': [0.98, 2.81, 4.93, 9.53, 24.19, 42.92, 65.56],
        'Erec (mJ)': [0.29, 0.96, 1.93, 2.95, 4.64, 5.49, 4.64]
    })
    edited_dp = st.data_editor(dp_df, num_rows="dynamic", key="dp_table")

with col_data2:
    st.subheader("📉 规格书通态压降矩阵 (V)")
    # Vce vs Ic 在不同结温下的表现
    vce_df = pd.DataFrame({
        'Ic (A)': [10, 100, 450, 600, 900],
        'Vce_25C (V)': [0.85, 1.10, 1.80, 2.20, 2.90],
        'Vce_150C (V)': [0.75, 1.05, 1.95, 2.50, 3.40]
    })
    edited_vce = st.data_editor(vce_df, num_rows="dynamic", key="vce_table")

# --- 2. 修正系数与环境设置 ---
st.divider()
st.header("2. 物理修正系数与工况设置")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**工况基本参数**")
    vdc = st.number_input("直流电压 Vdc (V)", value=510.0)
    v_ref = st.number_input("双脉冲基准 Vref (V)", value=510.0)
    iout = st.number_input("电流有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)

with c2:
    st.markdown("**调制与电阻**")
    m_index = st.number_input("调制度 M", value=0.92)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)
    fout = st.number_input("输出频率 fout (Hz)", value=50.0)
    rg_ext = st.number_input("外部驱动电阻 Rg (Ω)", value=5.0)
    rg_ref = st.number_input("基准电阻 Rg_ref (Ω)", value=5.0)

with c3:
    st.markdown("**修正指数 (重要)**")
    kv_exp = st.number_input("电压修正指数 Kv", value=1.3, help="Esw 随电压变化的幂次")
    ki_exp = st.number_input("电流修正指数 Ki", value=1.0, help="IGBT通常取1.0，SiC取0.6左右")
    kr_factor = st.number_input("电阻修正系数 Kr", value=1.0, help="Rg 对损耗的影响倍率")

with c4:
    st.markdown("**热阻与环境**")
    rth_jc = st.number_input("结到壳热阻 RthJC (K/W)", value=0.065, format="%.4f")
    t_case = st.number_input("基板/外壳温度 Tc (℃)", value=65.0)
    mode = st.selectbox("调制模式", ["SVPWM", "PWM"])

# --- 3. 核心计算逻辑 (动态插值 + 闭环迭代) ---
def run_solver():
    # 建立插值函数
    f_eon = interp1d(edited_dp['Ic (A)'], edited_dp['Eon (mJ)'], kind='quadratic', fill_value="extrapolate")
    f_eoff = interp1d(edited_dp['Ic (A)'], edited_dp['Eoff (mJ)'], kind='quadratic', fill_value="extrapolate")
    f_erec = interp1d(edited_dp['Ic (A)'], edited_dp['Erec (mJ)'], kind='quadratic', fill_value="extrapolate")
    
    f_vce_25 = interp1d(edited_vce['Ic (A)'], edited_vce['Vce_25C (V)'], kind='quadratic', fill_value="extrapolate")
    f_vce_150 = interp1d(edited_vce['Ic (A)'], edited_vce['Vce_150C (V)'], kind='quadratic', fill_value="extrapolate")

    # 初始结温猜测
    tj_loop = t_case + 10.0
    i_pk = math.sqrt(2) * iout
    
    # 闭环迭代算法
    for i in range(10):
        # 1. 开关损耗计算 (基于当前 Tj 修正)
        e_sw_base = f_eon(iout) + f_eoff(iout) + f_erec(iout)
        v_corr = math.pow(vdc / v_ref, kv_exp)
        # 温度对开关能量的修正 (常见公式: 1 + 0.003*(Tj-25))
        t_corr_sw = 1 + 0.003 * (tj_loop - 25.0)
        p_sw = (1/math.pi) * fsw * (e_sw_base / 1000) * v_corr * t_corr_sw * kr_factor

        # 2. 导通损耗计算 (双结温插值确定 Tj 下的 Vce)
        vce_at_tj = f_vce_25(iout) + (f_vce_150(iout) - f_vce_25(iout)) * (tj_loop - 25.0) / (150.0 - 25.0)
        
        phi = math.acos(cosphi)
        if mode == "SVPWM":
            k_v0 = 0.25 * m_index * cosphi
            k_r = (24*cosphi - 2*math.sqrt(3)*math.cos(2*phi) - 3*math.sqrt(3)) / 24
        else:
            k_v0 = (1 / (2 * math.pi)) + (m_index * cosphi / 8)
            k_r = (1 / 8) + (m_index * cosphi / (3 * math.pi))
        
        # 简化导通功耗估算 (结合 M 和 cosphi)
        p_cond = vce_at_tj * iout * (k_v0 * 4 + k_r * 2) / 2 # 解析项加权

        p_total = p_sw + p_cond
        
        # 3. 结温反馈 Tj = Tc + P * Rth
        tj_new = t_case + p_total * rth_jc
        
        if abs(tj_new - tj_loop) < 0.05:
            break
        tj_loop = tj_new

    return p_cond, p_sw, tj_loop

if st.button("🚀 执行全工况电热闭环仿真"):
    p_cond, p_sw, tj_final = run_solver()
    
    st.divider()
    st.subheader("3. 仿真结果分析")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("稳定结温 Tj", f"{tj_final:.2f} ℃")
    r2.metric("总功耗 P_total", f"{p_cond + p_sw:.2f} W")
    r3.metric("导通损耗 P_cond", f"{p_cond:.2f} W")
    r4.metric("开关损耗 P_sw", f"{p_sw:.2f} W")

    st.markdown("### 🔍 数学模型透明化展示")
    st.latex(r"P_{total} = P_{cond}(I_{out}, T_j, M, \cos\phi) + P_{sw}(I_{out}, V_{dc}, T_j, R_g)")
    st.latex(r"T_j = T_c + P_{total} \times R_{th(jc)}")
    st.write("注：当前开关损耗已根据实测矩阵进行 **Quadratic (二次) 插值**。")
