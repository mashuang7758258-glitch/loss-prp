import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD模块电热耦合对标 Pro", layout="wide")
st.title("🛡️ 功率模块芯片级电热耦合对标平台 (Pro)")

# --- 1. 规格书多温度数据矩阵录入 ---
st.header("1. 多温度规格书矩阵 (Multi-Temp Datasheet)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📉 通态压降矩阵 Vce(I, T) [V]")
    # 增加多温度列
    vce_data = pd.DataFrame({
        'Ic (A)': [10.0, 100.0, 450.0, 600.0, 900.0],
        '25C': [0.85, 1.10, 1.80, 2.20, 2.90],
        '125C': [0.78, 1.07, 1.90, 2.40, 3.20],
        '150C': [0.75, 1.05, 1.95, 2.50, 3.40],
        '175C': [0.72, 1.02, 2.05, 2.65, 3.65]
    })
    evce = st.data_editor(vce_data, num_rows="dynamic", key="vce_pro")

with col_d2:
    st.subheader("⚡ 开关能量矩阵 Esw(I, T) [mJ]")
    # 记录 Eon + Eoff + Erec 的总和点
    esw_data = pd.DataFrame({
        'Ic (A)': [100.0, 450.0, 900.0],
        '25C': [2.15, 11.16, 28.79],
        '150C': [3.50, 18.50, 45.20],
        '175C': [3.90, 21.00, 51.50]
    })
    eesw = st.data_editor(esw_data, num_rows="dynamic", key="esw_pro")

# --- 2. 结构参数与门极电阻设置 ---
st.divider()
st.header("2. 芯片结构与门极驱动参数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**芯片与桥臂结构**")
    n_chips = st.number_input("桥臂并联芯片数 (N)", value=6)
    r_arm = st.number_input("桥臂/引脚内阻 (mΩ)", value=0.5) / 1000
    rth_jc = st.number_input("RthJC (K/W)", value=0.065, format="%.4f")

with c2:
    st.markdown("**双门极电阻设置**")
    rg_int = st.number_input("内部门极电阻 Rg_int (Ω)", value=2.0)
    rg_on_ext = st.number_input("开通电阻 Rg_on_ext (Ω)", value=5.0)
    rg_off_ext = st.number_input("关断电阻 Rg_off_ext (Ω)", value=25.0)
    rg_ref = st.number_input("测试基准电阻 Rg_ref (Ω)", value=5.0)

with c3:
    st.markdown("**实时运行工况**")
    vdc = st.number_input("母线电压 Vdc (V)", value=780.0)
    iout = st.number_input("电流 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    cosphi = st.number_input("功率因数 cosφ", value=0.92)

with c4:
    st.markdown("**调制与修正**")
    m_index = st.number_input("调制度 M", value=0.92)
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)
    mode = st.selectbox("调制模式", ["SVPWM", "PWM"])
    kv_exp = st.number_input("电压修正指数 Kv", value=1.3)

# --- 3. 核心计算引擎 (多维插值迭代) ---
def multi_temp_interp(df, target_i, target_t, label_prefix=""):
    """执行 2D 插值：先按电流插值，再按温度线性外推"""
    temps = [int(col.replace('C', '')) for col in df.columns if 'C' in col]
    # 对每个温度列进行电流维度的插值
    vals_at_i = []
    for temp in temps:
        col_name = f"{temp}C"
        f = interp1d(df['Ic (A)'], df[col_name], kind='linear', fill_value="extrapolate")
        vals_at_i.append(float(f(target_i)))
    
    # 针对目标结温 Tj 进行温度维度的插值
    f_t = interp1d(temps, vals_at_i, kind='linear', fill_value="extrapolate")
    return float(f_t(target_t))

if st.button("🚀 执行芯片级电热闭环仿真"):
    tj_loop = t_case + 5.0
    i_chip = iout / n_chips # 芯片级电流分摊
    
    # 迭代对齐损耗与结温
    for i in range(12):
        # A. 导通损耗 (考虑芯片 Vce + 桥臂内阻)
        vce_chip = multi_temp_interp(evce, iout, tj_loop)
        v_total = vce_chip + iout * r_arm
        
        phi = math.acos(cosphi)
        if mode == "SVPWM":
            k_v0, k_r = 0.25*m_index*cosphi, (24*cosphi - 2*math.sqrt(3)*math.cos(2*phi) - 3*math.sqrt(3))/24
        else:
            k_v0, k_r = (1/(2*math.pi)) + (m_index*cosphi/8), (1/8) + (m_index*cosphi/(3*math.pi))
        
        p_cond = v_total * iout * (k_v0 * 4 + k_r * 2) / 2

        # B. 开关损耗 (考虑双电阻修正)
        esw_base = multi_temp_interp(eesw, iout, tj_loop)
        # 电阻修正逻辑: Rg_total / Rg_ref (简化模型，可按需改为幂函数)
        rg_on_total = rg_int + rg_on_ext
        rg_off_total = rg_int + rg_off_ext
        r_corr = ((rg_on_total + rg_off_total) / 2) / rg_ref
        
        v_corr = math.pow(vdc / 510.0, kv_exp) # 假设 510V 为测试基准
        p_sw = (1/math.pi) * fsw * (esw_base / 1000) * v_corr * r_corr

        p_total = p_sw + p_cond
        tj_new = t_case + p_total * rth_jc
        
        if abs(tj_new - tj_loop) < 0.05: break
        tj_loop = tj_new

    # --- 结果展示 ---
    st.divider()
    res1, res2, res3, res4 = st.columns(4)
    res1.metric("计算结温 Tj", f"{tj_loop:.2f} ℃")
    res2.metric("芯片级电流 (I/N)", f"{i_chip:.2f} A")
    res3.metric("导通功耗 P_cond", f"{p_cond:.2f} W")
    res4.metric("开关功耗 P_sw", f"{p_sw:.2f} W")
    
    st.info(f"对标提醒：当前 Rth={rth_jc}，若实测 Tj 偏高，请检查 Rg_off({rg_off_total}Ω) 导致的关断拖尾损耗。")
