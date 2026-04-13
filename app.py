import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD双臂独立仿真平台", layout="wide")
st.title("🛡️ 半桥模块上/下桥臂独立电热耦合对标平台")

# --- 1. 独立录入：晶体管 (T) 与 二极管 (D) 的数据 ---
st.header("1. 芯片特性矩阵 (T & D)")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("📊 晶体管特性 (T1/T2)")
    # Vce 和 Esw (Eon+Eoff)
    vce_data = pd.DataFrame({
        'Ic (A)': [100.0, 450.0, 900.0],
        '25C_V': [1.10, 1.80, 2.90], '150C_V': [1.05, 1.95, 3.40],
        '25C_E': [5.0, 25.0, 65.0],  '150C_E': [8.0, 42.0, 110.0]
    })
    evce = st.data_editor(vce_data, num_rows="dynamic", key="vce_t")

with col_d2:
    st.subheader("📉 二极管特性 (D1/D2)")
    # Vf 和 Erec
    vf_data = pd.DataFrame({
        'If (A)': [100.0, 450.0, 900.0],
        '25C_V': [1.20, 1.70, 2.30], '150C_V': [1.10, 1.85, 2.60],
        '25C_E': [1.50, 6.00, 15.0], '150C_E': [3.00, 12.0, 30.0]
    })
    evf = st.data_editor(vf_data, num_rows="dynamic", key="vf_d")

# --- 2. 桥臂差异参数设置 ---
st.divider()
st.header("2. 桥臂结构与驱动差异")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**结构差异 (散热/内阻)**")
    n_chips = st.number_input("每臂并联芯片数", value=6)
    r_arm_up = st.number_input("上桥臂内阻 (mΩ)", value=0.55) / 1000
    r_arm_low = st.number_input("下桥臂内阻 (mΩ)", value=0.48) / 1000
    rth_jc = st.number_input("芯片 RthJC (K/W)", value=0.065, format="%.4f")

with c2:
    st.markdown("**驱动电阻差异**")
    rg_on = st.number_input("Rg_on (Ω)", value=5.0)
    rg_off = st.number_input("Rg_off (Ω)", value=10.0)
    t_dead = st.number_input("死区时间 Deadtime (us)", value=2.5) / 1e6

with c3:
    st.markdown("**运行工况**")
    vdc = st.number_input("Vdc (V)", value=780.0)
    iout = st.number_input("Iout (Arms)", value=187.0)
    fsw = st.number_input("fsw (Hz)", value=10000)
    cosphi = st.number_input("cosφ", value=0.92)

with c4:
    st.markdown("**调制参数**")
    m_index = st.number_input("调制度 M", value=0.92)
    t_case = st.number_input("基板温度 Tc (℃)", value=65.0)
    mode = st.selectbox("调制模式", ["SVPWM", "PWM"])

# --- 3. 核心计算引擎 (双臂解耦迭代) ---
def run_bridge_solver():
    # 建立插值函数 (T 和 D 分开)
    f_vce_25 = interp1d(evce['Ic (A)'], evce['25C_V'], kind='linear', fill_value="extrapolate")
    f_vce_150 = interp1d(evce['Ic (A)'], evce['150C_V'], kind='linear', fill_value="extrapolate")
    f_esw_25 = interp1d(evce['Ic (A)'], evce['25C_E'], kind='linear', fill_value="extrapolate")
    f_esw_150 = interp1d(evce['Ic (A)'], evce['150C_E'], kind='linear', fill_value="extrapolate")

    f_vf_25 = interp1d(evf['If (A)'], evf['25C_V'], kind='linear', fill_value="extrapolate")
    f_vf_150 = interp1d(evf['If (A)'], evf['150C_V'], kind='linear', fill_value="extrapolate")
    f_erec_25 = interp1d(evf['If (A)'], evf['25C_E'], kind='linear', fill_value="extrapolate")
    f_erec_150 = interp1d(evf['If (A)'], evf['150C_E'], kind='linear', fill_value="extrapolate")

    # 初始结温：上桥 Tj_up, 下桥 Tj_low
    tj_up, tj_low = t_case + 5.0, t_case + 5.0
    phi = math.acos(cosphi)
    
    # 迭代对齐
    for i in range(15):
        # 调制占空比修正项 (基于 SVPWM/PWM 解析公式)
        if mode == "SVPWM":
            d_t = 1/2 + m_index*cosphi/2 # 简化版：T 管有效导通率
            d_d = 1/2 - m_index*cosphi/2 # 简化版：D 管有效导通率
        else:
            d_t = (1/math.pi) + (m_index*cosphi/4)
            d_d = (1/math.pi) - (m_index*cosphi/4)

        # --- A. 上桥计算 ---
        vce_up = f_vce_25(iout) + (f_vce_150(iout)-f_vce_25(iout))*(tj_up-25)/125
        p_cond_up = (vce_up + iout * r_arm_up) * iout * d_t
        esw_up = f_esw_25(iout) + (f_esw_150(iout)-f_esw_25(iout))*(tj_up-25)/125
        p_sw_up = (1/math.pi) * fsw * (esw_up/1000) * (vdc/510.0)**1.3 * (rg_on/5.0)
        
        # --- B. 下桥计算 ---
        vce_low = f_vce_25(iout) + (f_vce_150(iout)-f_vce_25(iout))*(tj_low-25)/125
        p_cond_low = (vce_low + iout * r_arm_low) * iout * d_t
        esw_low = f_esw_25(iout) + (f_esw_150(iout)-f_esw_25(iout))*(tj_low-25)/125
        p_sw_low = (1/math.pi) * fsw * (esw_low/1000) * (vdc/510.0)**1.3 * (rg_on/5.0)

        # C. 结温反馈
        tj_up_new = t_case + (p_cond_up + p_sw_up) * rth_jc
        tj_low_new = t_case + (p_cond_low + p_sw_low) * rth_jc
        
        if abs(tj_up_new - tj_up) < 0.05 and abs(tj_low_new - tj_low) < 0.05:
            break
        tj_up, tj_low = tj_up_new, tj_low_new

    return p_cond_up, p_sw_up, tj_up, p_cond_low, p_sw_low, tj_low

if st.button("🚀 执行双桥臂独立仿真"):
    pcu, psu, tju, pcl, psl, tjl = run_bridge_solver()
    
    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        st.subheader("🔼 上桥臂 (Upper Arm)")
        st.metric("结温 Tj_up", f"{tju:.2f} ℃")
        st.write(f"导通: {pcu:.1f}W | 开关: {psu:.1f}W")
    with res2:
        st.subheader("🔽 下桥臂 (Lower Arm)")
        st.metric("结温 Tj_low", f"{tjl:.2f} ℃")
        st.write(f"导通: {pcl:.1f}W | 开关: {psl:.1f}W")
    
    st.warning(f"分析：两臂温差为 {abs(tju-tjl):.2f} K。这主要源于内阻差异 R_up={r_arm_up*1000}mΩ vs R_low={r_arm_low*1000}mΩ。")
