import io
import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-物理修复版", layout="wide")

DEFAULT_MAIN_VI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "V_drop (V)": [1.10, 1.05, 2.20, 2.50]})
DEFAULT_MAIN_EI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Eon (mJ)": [5.9, 8.5, 70.0, 95.0], "Eoff (mJ)": [4.9, 7.2, 45.0, 60.0]})
DEFAULT_DIODE_VI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Vf (V)": [1.20, 1.10, 2.00, 2.20]})
DEFAULT_DIODE_EI = pd.DataFrame({"Temp (℃)": [25, 150, 25, 150], "Current (A)": [100.0, 100.0, 600.0, 600.0], "Erec (mJ)": [1.9, 3.5, 15.0, 25.0]})

TEMP_COL = "Temp (degC)"
CURRENT_COL = "Current (A)"

def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))

def canonicalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        col_str = str(col)
        if "Temp" in col_str or "temp" in col_str: rename_map[col] = TEMP_COL
        elif "Current" in col_str or col_str.strip().lower() in {"ic (a)", "if (a)", "current"}: rename_map[col] = CURRENT_COL
    return df.rename(columns=rename_map)

def validate_numeric_table(df: pd.DataFrame, table_name: str, required_cols: list[str]):
    cleaned = canonicalize_df_columns(df.copy()).replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    missing_cols = [col for col in required_cols if col not in cleaned.columns]
    if missing_cols: return cleaned, [f"{table_name} 缺少必要列：{', '.join(missing_cols)}"], []
    raw_required = cleaned[required_cols].copy()
    numeric_required = raw_required.apply(pd.to_numeric, errors="coerce")
    cleaned[required_cols] = numeric_required
    cleaned = cleaned.dropna(subset=required_cols).sort_values([TEMP_COL, CURRENT_COL]).reset_index(drop=True)
    return cleaned, [], []

def describe_temperature_strategy(df: pd.DataFrame, temp_coeff: float) -> dict:
    unique_temps = sorted(float(t) for t in df[TEMP_COL].dropna().unique()) if TEMP_COL in df.columns else []
    multi_temp = len(unique_temps) >= 2
    return {"multi_temp": multi_temp, "effective_temp_coeff": 0.0 if multi_temp else float(temp_coeff), "strategy_label": "二维曲面插值" if multi_temp else "温漂外推"}

def normalize_vi_df(df: pd.DataFrame, n_src: int) -> pd.DataFrame:
    res_df = df.copy()
    if n_src > 1: res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
    return res_df

def normalize_ei_df(df: pd.DataFrame, n_src: int, e_cols: list[str]) -> pd.DataFrame:
    res_df = df.copy()
    if n_src > 1:
        res_df[CURRENT_COL] = res_df[CURRENT_COL] / float(n_src)
        for col in e_cols:
            if col in res_df.columns: res_df[col] = res_df[col] / float(n_src)
    return res_df

def safe_interp(df: pd.DataFrame, target_i: float, target_t: float, item_name: str) -> float:
    clean_df = canonicalize_df_columns(df.dropna())
    if clean_df.empty or item_name not in clean_df.columns: return 0.0
    temp_list, val_list = [], []
    for temp, group in clean_df.groupby(TEMP_COL):
        sorted_group = group.sort_values(CURRENT_COL)
        if len(sorted_group) >= 2:
            val_list.append(max(0.0, float(interp1d(sorted_group[CURRENT_COL], sorted_group[item_name], kind="linear", fill_value="extrapolate")(target_i))))
            temp_list.append(float(temp))
        elif len(sorted_group) == 1:
            val_list.append(max(0.0, float(sorted_group[item_name].iloc[0])))
            temp_list.append(float(temp))
    if len(temp_list) >= 2: return max(0.0, float(interp1d(temp_list, val_list, kind="linear", fill_value="extrapolate")(target_t)))
    elif len(temp_list) == 1: return max(0.0, float(val_list[0]))
    return 0.0

def build_linearized_device_model(df: pd.DataFrame, target_i: float, target_t: float, item_name: str, force_zero_intercept: bool, ext_mode: str, i_nom_chip: float):
    """修复2：支持公司静态标称点提取法"""
    if "静态" in ext_mode:
        i_high = max(float(i_nom_chip), 1e-6)
        i_low = max(float(i_nom_chip) / 2.0, 1e-6)
    else:
        i_high = max(float(target_i), 1e-6)
        i_low = max(target_i / 2.0, 1e-6)

    v_pk = safe_interp(df, i_high, target_t, item_name)
    v_half = safe_interp(df, i_low, target_t, item_name)
    denom = i_high - i_low
    r_eq = max(0.0, (v_pk - v_half) / denom) if denom > 1e-12 else 0.0
    v0 = 0.0 if force_zero_intercept else max(0.0, v_pk - r_eq * i_high)
    return {"v_pk": v_pk, "v_half": v_half, "r_eq": r_eq, "v0": v0}

def calc_switching_energy(df: pd.DataFrame, i_pk: float, tj: float, algo_type: str, i_nom_chip: float, item_name: str, vdc: float, vref: float, kv: float, ract: float, rref: float, kr: float, temp_coeff: float, tref: float) -> dict:
    """修复3：解耦单芯与模块的电流映射"""
    strategy_meta = describe_temperature_strategy(df, temp_coeff)
    if "直线" in algo_type:
        nominal_curr = max(float(i_nom_chip), 1e-12)
        e_nom = safe_interp(df, nominal_curr, tj, item_name)
        e_base = e_nom * (max(float(i_pk), 0.0) / nominal_curr)
    else:
        e_base = safe_interp(df, i_pk, tj, item_name)
        
    temp_correction = 1.0 if strategy_meta["multi_temp"] else max(0.0, 1.0 + strategy_meta["effective_temp_coeff"] * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    return {"energy_mj": max(0.0, e_base * temp_correction * rg_correction * voltage_correction)}

def calc_pwm_conduction_losses(mode: str, m_eff: float, active_cosphi: float, theta: float, i_pk_chip: float, main_model: dict, diode_model: dict, r_pkg_chip: float, r_arm_chip: float, dead_ratio: float, current_sign: float):
    """修复1：重塑纯正的 SVPWM 物理积分公式"""
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    if mode == "SVPWM":
        # 补充被 Codex 弄丢的 1/(2pi) 和 1/8 基础直流项！
        kv0_m = 1.0 / (2.0 * math.pi) + (m_eff * active_cosphi) / 8.0
        kr_m = 1.0 / 8.0 + m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / (24.0 * math.pi)
        kv0_d = 1.0 / (2.0 * math.pi) - (m_eff * active_cosphi) / 8.0
        kr_d = 1.0 / 8.0 + m_eff * (6.0 * math.pi - 24.0 * active_cosphi + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) + 3.0 * math.sqrt(3.0)) / (24.0 * math.pi)
        
        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2)
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) + (kr_d * r_diode_total * i_pk_chip**2)
    else:
        p_cond_main = main_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0) + r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        p_cond_diode = diode_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0) + r_diode_total * i_pk_chip**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))

    # 死区补偿修复：使用半波等效平均值，而非瞬态峰值扣减
    if dead_ratio > 0.0:
        avg_inst_main = (main_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_main_total * i_pk_chip**2 * 0.5)
        avg_inst_diode = (diode_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_diode_total * i_pk_chip**2 * 0.5)
        if current_sign >= 0.0:
            p_cond_main = max(0.0, p_cond_main - dead_ratio * avg_inst_main)
            p_cond_diode += dead_ratio * avg_inst_diode
        else:
            p_cond_diode = max(0.0, p_cond_diode - dead_ratio * avg_inst_diode)
            p_cond_main += dead_ratio * avg_inst_main

    return {"p_cond_main": max(0.0, p_cond_main), "p_cond_diode": max(0.0, p_cond_diode)}

# ================= UI 渲染部分 =================
with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 芯片类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    st.divider()
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=2, min_value=1)
    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=2, min_value=1)
    n_sim = st.number_input("目标仿真单臂芯片数", value=2, min_value=1)
    sim_mode = st.radio("模式选择", ["A. 开环盲算", "B. 闭环迭代"])
    fixed_tj = st.number_input("设定全局结温 (℃)", value=150.0)

st.header("第一步：数据录入")
col_main, col_diode = st.columns(2)
with col_main:
    ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
    ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")
with col_diode:
    ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
    ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

st.header("第二步：参数配置与对标算法")
c1, c2, c3, c4 = st.columns(4)
with c1:
    op_mode = st.selectbox("运行场景", ["电动/巡航", "制动/反拖", "最恶劣堵转"])
    vdc_act = st.number_input("实际母线 V_dc", value=650.0)
    iout_rms = st.number_input("输出有效值 I_out", value=285.0)
    fsw = st.number_input("开关频率 f_sw", value=10000.0)
    mode = st.selectbox("调制模式", ["SVPWM", "SPWM"])
    cosphi = st.number_input("功率因数", value=0.90)
    m_index = st.number_input("调制系数 M", value=0.90)
with c2:
    algo_type = st.radio("开关能量提取", ["1. CAE精确二维插值", "2. 直线比例法(对标公司)"])
    i_nom_ref = st.number_input("直线基准电流(整臂)", value=400.0)
    v0_strategy = st.radio("V0/Req 提取", ["1. 动态自适应 (按Ipk)", "2. 静态标称点 (对标公司)"])
    v_ref = st.number_input("测试基准 V_nom", value=450.0)
    dead_time_us = st.number_input("死区时间 t_dead (us)", value=0.0)
with c3:
    kv_on = st.number_input("开通指数 K_v_on", value=1.30)
    kv_off = st.number_input("关断指数 K_v_off", value=1.30)
    kron = st.number_input("电阻系数 K_ron", value=0.30)
    kroff = st.number_input("电阻系数 K_roff", value=0.50)
with c4:
    st.info("对标公司报告时，请将温漂全部填 0")
    t_coeff_igbt = st.number_input("主管温漂系数", value=0.000, format="%.4f")
    t_coeff_frd = st.number_input("续流温漂系数", value=0.000, format="%.4f")
    rg_on_ref = st.number_input("R_on (Ω)", value=2.5)

if st.button("🚀 执 行 全 工 况 仿 真 计 算", use_container_width=True):
    norm_ev_m, norm_ev_d = normalize_vi_df(ev_main, n_src_cond), normalize_vi_df(ev_diode, n_src_cond)
    norm_ee_m, norm_ee_d = normalize_ei_df(ee_main, n_src_sw, ["Eon (mJ)", "Eoff (mJ)"]), normalize_ei_df(ee_diode, n_src_sw, ["Erec (mJ)"])

    i_pk_chip = math.sqrt(2.0) * (iout_rms / n_sim)
    i_nom_chip = i_nom_ref / n_src_sw if "模块" in sw_data_type else i_nom_ref
    
    active_cosphi = -cosphi if "反拖" in op_mode else cosphi
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0
    m_eff = m_index # 简化的死区逻辑，此处专注展示修复核心
    dead_ratio = 0.0

    tj_current = fixed_tj
    for _ in range(1 if "开环" in sim_mode else 15):
        main_model = build_linearized_device_model(norm_ev_m, i_pk_chip, tj_current, "V_drop (V)", "SiC" in device_type, v0_strategy, i_nom_chip)
        diode_model = build_linearized_device_model(norm_ev_d, i_pk_chip, tj_current, "Vf (V)", False, v0_strategy, i_nom_chip)

        if "堵转" in op_mode:
            d_max = 0.5 * (1.0 + m_eff)
            cond_res = {"p_cond_main": d_max * (main_model["v0"]*i_pk_chip + main_model["r_eq"]*i_pk_chip**2), "p_cond_diode": (1-d_max) * (diode_model["v0"]*i_pk_chip + diode_model["r_eq"]*i_pk_chip**2)}
            f_factor = fsw
        else:
            cond_res = calc_pwm_conduction_losses(mode, m_eff, active_cosphi, theta, i_pk_chip, main_model, diode_model, 0.0, 0.0, dead_ratio, 1.0)
            f_factor = fsw / math.pi

        eon = calc_switching_energy(norm_ee_m, i_pk_chip, tj_current, algo_type, i_nom_chip, "Eon (mJ)", vdc_act, v_ref, kv_on, rg_on_ref, rg_on_ref, kron, t_coeff_igbt, 25.0)["energy_mj"]
        eoff = calc_switching_energy(norm_ee_m, i_pk_chip, tj_current, algo_type, i_nom_chip, "Eoff (mJ)", vdc_act, v_ref, kv_off, rg_on_ref, rg_on_ref, kroff, t_coeff_igbt, 25.0)["energy_mj"]
        erec = calc_switching_energy(norm_ee_d, i_pk_chip, tj_current, algo_type, i_nom_chip, "Erec (mJ)", vdc_act, v_ref, 0.6, 1.0, 1.0, 0.0, t_coeff_frd, 25.0)["energy_mj"]

        p_sw_m_chip = f_factor * (eon + eoff) / 1000.0
        p_sw_d_chip = f_factor * erec / 1000.0

    st.success(f"✅ 计算完成！已应用物理级修正算法。")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("单颗峰值电流 I_pk", f"{i_pk_chip:.2f} A")
    r2.metric("主芯片系统总损耗", f"{(cond_res['p_cond_main'] + p_sw_m_chip)*n_sim:.1f} W")
    r3.metric("🔴 主芯片单颗发热率", f"{cond_res['p_cond_main'] + p_sw_m_chip:.2f} W")
    r4.metric("🔵 二极管单颗发热率", f"{cond_res['p_cond_diode'] + p_sw_d_chip:.2f} W")
    
    st.write(f"> 详细拆分：主导通 **{cond_res['p_cond_main']:.1f} W** | 主开关 **{p_sw_m_chip:.1f} W**")
