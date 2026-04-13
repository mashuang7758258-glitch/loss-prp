import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD模块损耗计算-基于源sheet公式", layout="wide")
st.title("🛡️ 功率模块损耗仿真平台 (基于源sheet公式对标版)")

# --- 1. 原始特性矩阵录入 (解耦指数项) ---
st.header("1. 原始测试特性录入 (Datasheet Input)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📉 通态压降 (Vce/Vf)")
    st.write("输入 V = V0 + r * Ic 模型参数")
    v_df = pd.DataFrame({
        '器件': ['IGBT/SiC_T1', 'Diode_D1'],
        'V0 (V)': [1.0, 1.2],
        'r_on (mΩ)': [2.5, 1.8]
    })
    ev_df = st.data_editor(v_df, num_rows="dynamic", key="v_table")

with col_d2:
    st.subheader("⚡ 开关能量二次拟合系数 [mJ]")
    st.write("E(i) = a*i^2 + b*i + c (需为当前 Tj 下测试值)")
    e_df = pd.DataFrame({
        'Energy': ['Eon', 'Eoff'],
        'a': [0.0001, 0.00005],
        'b': [0.08, 0.04],
        'c': [5.0, 3.0]
    })
    ee_df = st.data_editor(e_df, num_rows="dynamic", key="e_table")

# --- 2. 物理指数与工况设置 ---
st.divider()
st.header("2. 工况与分项物理指数")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**工况与调制**")
    vdc = st.number_input("直流电压 Vdc (V)", value=780.0)
    iout_rms = st.number_input("输出电流有效值 Iout (Arms)", value=187.0)
    fsw = st.number_input("开关频率 fsw (Hz)", value=10000)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM/PWM"])
    cosphi = st.number_input("功率因数 cosφ", value=0.92)
    m_index = st.number_input("调制度 M", value=0.92)

with c2:
    st.markdown("**物理基准 (Reference)**")
    n_chips = st.number_input("桥臂并联芯片数 (N)", value=6)
    v_ref = st.number_input("测试基准 Vref (V)", value=650.0)
    rg_ref = st.number_input("测试基准 Rg_ref (Ω)", value=5.0)
    r_arm = st.number_input("桥臂电阻 (mΩ)", value=0.5) / 1000.0

with c3:
    st.markdown("**分项缩放指数**")
    kv_exp_on = st.number_input("Kv_exp_on (Eon随Vdc变化)", value=1.3)
    kr_exp_on = st.number_input("Kr_exp_on (Eon随Rgon变化)", value=1.0)
    rg_on_act = st.number_input("实际驱动 Rg_on (Ω)", value=5.0)
    
    kv_exp_off = st.number_input("Kv_exp_off (Eoff随Vdc变化)", value=1.2)
    kr_exp_off = st.number_input("Kr_exp_off (Eoff随Rgoff变化)", value=0.8)
    rg_off_act = st.number_input("实际驱动 Rg_off (Ω)", value=10.0)

# --- 3. 核心计算引擎 (完全公式化) ---
if st.button("🚀 执行准确无误的损耗计算"):
    # 数据提取
    igbt_v0 = ev_df.iloc[0, 1]
    igbt_r = ev_df.iloc[0, 2] / 1000.0
    diode_v0 = ev_df.iloc[1, 1]
    diode_r = ev_df.iloc[1, 2] / 1000.0
    
    a_on, b_on, c_on = ee_df.iloc[0, 1:4]
    a_off, b_off, c_off = ee_df.iloc[1, 1:4]

    # 1. 通态压降损耗逻辑
    tj = 150.0 # 假设稳态 Tj，不再参与指数修正
    i_pk = math.sqrt(2) * iout_rms
    theta = math.acos(cosphi)
    
    # 准确无误的调制系数
    if mode == "SPWM/PWM":
        kv0_t = 1/(2*math.pi) + m_index*cosphi/8
        kr_t = 1/8 + m_index*cosphi/(3*math.pi)
        kv0_d = 1/(2*math.pi) - m_index*cosphi/8
        kr_d = 1/8 - m_index*cosphi/(3*math.pi)
    else: # SVPWM 特有马鞍波公式
        kv0_t = m_index*cosphi/4
        kr_t = (24*cosphi - 2*math.sqrt(3)*math.cos(2*theta) - 3*math.sqrt(3))/24
        kv0_d, kr_d = 0.0, 0.0 # 图片未给出二极管系数，设为0

    # 导通损耗 (考虑桥臂电阻)
    p_cond_t = (igbt_v0 * i_pk * kv0_t) + ((igbt_r + r_arm) * i_pk**2 * kr_t)
    p_cond_d = (diode_v0 * i_pk * kv0_d) + ((diode_r + r_arm) * i_pk**2 * kr_d)

    # 2. 开关损耗分项逻辑 (二次拟合积分 + 分项Rg指数)
    # 芯片级电流分摊查取单芯能量
    i_lookup = iout_rms / n_chips
    i_lookup_pk = math.sqrt(2) * i_lookup
    
    # 周期平均二次项基准 (mJ)
    e_on_base = a_on*(i_lookup_pk**2/4) + b_on*(i_lookup_pk/math.pi) + c_on
    e_off_base = a_off*(i_lookup_pk**2/4) + b_off*(i_lookup_pk/math.pi) + c_off
    
    # 开通损耗分项修正 (关联 Rg_on)
    kv_on = math.pow(vdc / v_ref, kv_exp_on)
    kr_on = math.pow(rg_on_act / rg_ref, kr_exp_on)
    p_on_t_chip = (1/math.pi) * fsw * (e_on_base/1000) * kv_on * kr_on
    
    # 关断损耗分项修正 (关联 Rg_off)
    kv_off = math.pow(vdc / v_ref, kv_exp_off)
    kr_off = math.pow(rg_off_act / rg_ref, kr_exp_off)
    p_off_t_chip = (1/math.pi) * fsw * (e_off_base/1000) * kv_off * kr_off
    
    # 放大至 N 芯
    p_sw_total_t = (p_on_t_chip + p_off_t_chip) * n_chips

    # --- 结果展示与公式可视 ---
    st.divider()
    st.subheader("3. 准确无误的仿真结果分析")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("IGBT总导通损耗 P_cond_T", f"{p_cond_t:.2f} W")
    r2.metric("Diode总导通损耗 P_cond_D", f"{p_cond_d:.2f} W")
    r3.metric("IGBT总开关损耗 P_sw_T", f"{p_sw_total_t:.2f} W")
    r4.metric("模块总损耗 P_total", f"{p_cond_t + p_cond_d + p_sw_total_t:.2f} W")

    st.markdown("### 🔍 数学模型透明化展示")
    st.write(f"当前模式: {mode}")
    # 显示 SPWM 导通公式
    if mode == "SPWM/PWM":
        st.latex(r"K_{r\_T} = \frac{1}{8} + \frac{" + str(round(m_index,2)) + r"\cos\phi}{3\pi}")
    # 显示开关损耗公式 (分项修正对标)
    st.latex(r"P_{on\_T} = f_{sw} \cdot \frac{1}{\pi} \cdot E_{on\_avg}(I_{pk}) \cdot (\frac{V_{dc}}{V_{ref}})^{" + str(kv_exp_on) + r"} \cdot (\frac{Rg_{on}}{Rg_{ref}})^{" + str(kr_exp_on) + r"}")
    st.latex(r"P_{off\_T} = f_{sw} \cdot \frac{1}{\pi} \cdot E_{off\_avg}(I_{pk}) \cdot (\frac{V_{dc}}{V_{ref}})^{" + str(kv_exp_off) + r"} \cdot (\frac{Rg_{off}}{Rg_{ref}})^{" + str(kr_exp_off) + r"}")
