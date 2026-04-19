import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d

st.set_page_config(page_title="BYD系统级电热仿真-物理对齐版", layout="wide")

# =============================================================================
# 默认填充数据
# =============================================================================
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
            func = interp1d(sorted_group[CURRENT_COL], sorted_group[item_name], kind="linear", fill_value="extrapolate")
            val_list.append(max(0.0, float(func(target_i))))
            temp_list.append(float(temp))
        elif len(sorted_group) == 1:
            val_list.append(max(0.0, float(sorted_group[item_name].iloc[0])))
            temp_list.append(float(temp))

    if len(temp_list) >= 2: return max(0.0, float(interp1d(temp_list, val_list, kind="linear", fill_value="extrapolate")(target_t)))
    elif len(temp_list) == 1: return max(0.0, float(val_list[0]))
    return 0.0

# ================= 核心修复：根据邻近点动态拟合 =================
def get_bracketing_points(i_list, target_i):
    """找到包含 target_i 的最近两个表电流点"""
    i_list = sorted(list(set(i_list)))
    if len(i_list) < 2:
        return (i_list[0], i_list[0]) if i_list else (1e-6, 1e-6)
    if target_i <= i_list[0]:
        return i_list[0], i_list[1]
    if target_i >= i_list[-1]:
        return i_list[-2], i_list[-1]
    for k in range(len(i_list)-1):
        if i_list[k] <= target_i <= i_list[k+1]:
            return i_list[k], i_list[k+1]
    return i_list[0], i_list[1]

def build_linearized_device_model(df: pd.DataFrame, target_i: float, target_t: float, item_name: str, force_zero_intercept: bool):
    """
    【完全对标要求】：不再使用 0 和 Ipk 拉直线，而是寻找最近的两个表电流来计算 Req 和 V0
    """
    clean_df = canonicalize_df_columns(df.dropna())
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    # 动态获取距离 Iout(Ipk) 最近的拟合点区间
    i_low, i_high = get_bracketing_points(i_list, target_i)

    v_low = safe_interp(df, i_low, target_t, item_name)
    v_high = safe_interp(df, i_high, target_t, item_name)

    denom = i_high - i_low
    r_eq = max(0.0, (v_high - v_low) / denom) if denom > 1e-12 else 0.0
    v0 = 0.0 if force_zero_intercept else max(0.0, v_low - r_eq * i_low)

    return {"v_pk": v_high, "v_half": v_low, "r_eq": r_eq, "v0": v0, "i_low": i_low, "i_high": i_high}

def calc_switching_energy(df: pd.DataFrame, i_pk: float, tj: float, algo_type: str, item_name: str, vdc: float, vref: float, kv: float, ract: float, rref: float, kr: float, temp_coeff: float, tref: float) -> dict:
    """
    【强制温漂逻辑】：解除一切智能锁，只要 temp_coeff > 0，就必定执行温漂公式
    """
    clean_df = canonicalize_df_columns(df.dropna())
    i_list = clean_df[CURRENT_COL].unique() if not clean_df.empty else []
    
    if "比例法" in algo_type and len(i_list) > 0:
        # 寻找最近的一个点按比例放缩
        i_closest = min(i_list, key=lambda x: abs(x - i_pk))
        e_nom = safe_interp(df, i_closest, tj, item_name)
        e_base = e_nom * (max(float(i_pk), 0.0) / max(i_closest, 1e-6))
        extraction_label = f"最近点比例法 (基准 {i_closest:.1f}A)"
        e_nom_mj = e_nom
    else:
        # CAE两点线性拟合（自带了最近点特征）
        e_base = safe_interp(df, i_pk, tj, item_name)
        i_low, i_high = get_bracketing_points(i_list, i_pk)
        extraction_label = f"就近两点插值 ({i_low:.1f}A~{i_high:.1f}A)"
        e_nom_mj = np.nan

    # === 强行执行温漂放大 ===
    temp_correction = max(0.0, 1.0 + float(temp_coeff) * (tj - tref))
    rg_correction = math.pow(max(ract, 1e-12) / max(rref, 1e-12), kr) if rref > 0 else 1.0
    voltage_correction = math.pow(max(vdc, 1e-12) / max(vref, 1e-12), kv) if vref > 0 else 1.0
    energy_mj = max(0.0, e_base * temp_correction * rg_correction * voltage_correction)

    return {"energy_mj": energy_mj, "e_base_mj": max(0.0, float(e_base)), "e_nom_mj": e_nom_mj, "temp_correction": temp_correction, "rg_correction": rg_correction, "voltage_correction": voltage_correction, "effective_temp_coeff": float(temp_coeff), "extraction_label": extraction_label}
# ================================================================

def calc_dead_time_compensation(mode: str, fsw: float, dead_time_us: float, m_index: float, current_sign: float, vdc: float):
    if fsw <= 0.0 or dead_time_us <= 0.0: return {"dead_ratio": 0.0, "m_eff": m_index, "current_sign": current_sign, "phase_voltage_error_v": 0.0, "modulation_gain": 0.0}
    dead_time_s = dead_time_us * 1e-6
    dead_ratio = clamp(2.0 * dead_time_s * fsw, 0.0, 0.20)
    modulation_gain = 4.0 / math.pi if mode == "SVPWM" else 2.0 / math.pi
    m_eff = clamp(m_index - current_sign * modulation_gain * dead_ratio, 0.0, 1.15)
    return {"dead_ratio": dead_ratio, "m_eff": m_eff, "current_sign": current_sign, "phase_voltage_error_v": 0.5 * vdc * dead_ratio * current_sign, "modulation_gain": modulation_gain}

def calc_pwm_conduction_losses(mode: str, m_eff: float, active_cosphi: float, theta: float, i_pk_chip: float, main_model: dict, diode_model: dict, r_pkg_chip: float, r_arm_chip: float, dead_meta: dict):
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    if mode == "SVPWM":
        kv0_m = (m_eff * active_cosphi) / 4.0
        kr_m = m_eff * (24.0 * active_cosphi - 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) - 3.0 * math.sqrt(3.0)) / 24.0
        kv0_d = (4.0 - m_eff * math.pi * active_cosphi) / 4.0
        kr_d = m_eff * (6.0 * math.pi - 24.0 * active_cosphi + 2.0 * math.sqrt(3.0) * math.cos(2.0 * theta) + 3.0 * math.sqrt(3.0)) / 24.0
        p_cond_main = (kv0_m * main_model["v0"] * i_pk_chip) + (kr_m * r_main_total * i_pk_chip**2) / math.pi
        p_cond_diode = (kv0_d * diode_model["v0"] * i_pk_chip) / math.pi + (kr_d * r_diode_total * i_pk_chip**2) / math.pi
    else:
        p_cond_main = main_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) + m_eff * active_cosphi / 8.0) + r_main_total * i_pk_chip**2 * (1.0 / 8.0 + m_eff * active_cosphi / (3.0 * math.pi))
        p_cond_diode = diode_model["v0"] * i_pk_chip * (1.0 / (2.0 * math.pi) - m_eff * active_cosphi / 8.0) + r_diode_total * i_pk_chip**2 * (1.0 / 8.0 - m_eff * active_cosphi / (3.0 * math.pi))

    p_cond_main = max(0.0, float(p_cond_main))
    p_cond_diode = max(0.0, float(p_cond_diode))

    dead_ratio = dead_meta["dead_ratio"]
    if dead_ratio > 0.0:
        avg_inst_main = (main_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_main_total * i_pk_chip**2 * 0.5)
        avg_inst_diode = (diode_model["v0"] * i_pk_chip * 2.0 / math.pi) + (r_diode_total * i_pk_chip**2 * 0.5)
        if dead_meta["current_sign"] >= 0.0:
            p_cond_main = max(0.0, p_cond_main - dead_ratio * avg_inst_main)
            p_cond_diode += dead_ratio * avg_inst_diode
        else:
            p_cond_diode = max(0.0, p_cond_diode - dead_ratio * avg_inst_diode)
            p_cond_main += dead_ratio * avg_inst_main

    return {"p_cond_main": p_cond_main, "p_cond_diode": p_cond_diode, "r_main_total": r_main_total, "r_diode_total": r_diode_total}

def calc_stall_losses(m_eff: float, i_pk_chip: float, main_model: dict, diode_model: dict, r_pkg_chip: float, r_arm_chip: float, dead_meta: dict):
    r_main_total = main_model["r_eq"] + r_pkg_chip + r_arm_chip
    r_diode_total = diode_model["r_eq"] + r_pkg_chip + r_arm_chip

    inst_main = main_model["v0"] * i_pk_chip + r_main_total * i_pk_chip**2
    inst_diode = diode_model["v0"] * i_pk_chip + r_diode_total * i_pk_chip**2
    d_max = clamp(0.5 * (1.0 + m_eff), 0.0, 1.0)
    p_cond_main = d_max * inst_main
    p_cond_diode = (1.0 - d_max) * inst_diode

    dead_ratio = dead_meta["dead_ratio"]
    if dead_ratio > 0.0:
        p_cond_main = max(0.0, p_cond_main - dead_ratio * inst_main)
        p_cond_diode += dead_ratio * inst_diode

    return {"p_cond_main": p_cond_main, "p_cond_diode": p_cond_diode, "r_main_total": r_main_total, "r_diode_total": r_diode_total, "d_max": d_max}

def simulate_system(inputs: dict, tables: dict):
    cond_src_count = inputs["n_src_cond"] if "Module" in inputs["cond_data_type"] else 1
    sw_src_count = inputs["n_src_sw"] if "Module" in inputs["sw_data_type"] else 1

    norm_ev_m = normalize_vi_df(tables["ev_main"], cond_src_count)
    norm_ev_d = normalize_vi_df(tables["ev_diode"], cond_src_count)
    norm_ee_m = normalize_ei_df(tables["ee_main"], sw_src_count, ["Eon (mJ)", "Eoff (mJ)"])
    norm_ee_d = normalize_ei_df(tables["ee_diode"], sw_src_count, ["Erec (mJ)"])

    i_pk_chip = math.sqrt(2.0) * (inputs["iout_rms"] / inputs["n_sim"]) if inputs["n_sim"] > 0 else 0.0
    r_arm_chip = (inputs["r_arm_mohm"] / 1000.0) * inputs["n_sim"]
    r_pkg_chip = inputs["r_pkg_mohm"] / 1000.0

    active_cosphi = -abs(inputs["cosphi"]) if "Regeneration" in inputs["op_mode"] else abs(inputs["cosphi"])
    active_cosphi = clamp(active_cosphi, -1.0, 1.0)
    theta = math.acos(active_cosphi) if abs(active_cosphi) <= 1.0 else 0.0
    current_sign = -1.0 if "Regeneration" in inputs["op_mode"] else 1.0
    dead_meta = calc_dead_time_compensation(inputs["mode"], inputs["fsw"], inputs["dead_time_us"], inputs["m_index"], current_sign, inputs["vdc_act"])

    max_iter = 30
    min_iter_before_break = 15 if "闭环" in inputs["sim_mode"] else 1
    tolerance = 0.05
    if "开环" in inputs["sim_mode"]:
        tj_main_current, tj_diode_current = inputs["fixed_tj"], inputs["fixed_tj"]
        loop_count = 1
    else:
        tj_main_current, tj_diode_current = inputs["t_case_main"] + 5.0, inputs["t_case_diode"] + 5.0
        loop_count = max_iter

    iteration_rows = []

    for loop_idx in range(loop_count):
        # 核心挂载点：利用邻近参数拟合
        main_model = build_linearized_device_model(norm_ev_m, i_pk_chip, tj_main_current, "V_drop (V)", force_zero_intercept="SiC" in inputs["device_type"])
        diode_model = build_linearized_device_model(norm_ev_d, i_pk_chip, tj_diode_current, "Vf (V)", force_zero_intercept=False)

        if "Stall" in inputs["op_mode"]:
            cond_result = calc_stall_losses(dead_meta["m_eff"], i_pk_chip, main_model, diode_model, r_pkg_chip, r_arm_chip, dead_meta)
        else:
            cond_result = calc_pwm_conduction_losses(inputs["mode"], dead_meta["m_eff"], active_cosphi, theta, i_pk_chip, main_model, diode_model, r_pkg_chip, r_arm_chip, dead_meta)

        eon_meta = calc_switching_energy(norm_ee_m, i_pk_chip, tj_main_current, inputs["algo_type"], "Eon (mJ)", inputs["vdc_act"], inputs["v_ref"], inputs["kv_on"], inputs["rg_on_act"], inputs["rg_on_ref"], inputs["kron"], inputs["t_coeff_igbt"], inputs["t_ref_dp"])
        eoff_meta = calc_switching_energy(norm_ee_m, i_pk_chip, tj_main_current, inputs["algo_type"], "Eoff (mJ)", inputs["vdc_act"], inputs["v_ref"], inputs["kv_off"], inputs["rg_off_act"], inputs["rg_off_ref"], inputs["kroff"], inputs["t_coeff_igbt"], inputs["t_ref_dp"])
        erec_meta = calc_switching_energy(norm_ee_d, i_pk_chip, tj_diode_current, inputs["algo_type"], "Erec (mJ)", inputs["vdc_act"], inputs["v_ref"], inputs["kv_frd"], 1.0, 1.0, 0.0, inputs["t_coeff_frd"], inputs["t_ref_dp"])

        if "Stall" in inputs["op_mode"]:
            p_sw_main_chip = inputs["fsw"] * ((eon_meta["energy_mj"] + eoff_meta["energy_mj"]) / 1000.0)
            p_sw_diode_chip = inputs["fsw"] * (erec_meta["energy_mj"] / 1000.0)
        else:
            i_corr = math.pow(max(inputs["iout_rms"], 1e-12) / max((i_pk_chip * inputs["n_sim"]) / math.sqrt(2.0), 1e-12), inputs["ki_frd"]) if inputs["ki_frd"] > 0 else 1.0
            p_sw_main_chip = (inputs["fsw"] / math.pi) * ((eon_meta["energy_mj"] + eoff_meta["energy_mj"]) / 1000.0)
            p_sw_diode_chip = (inputs["fsw"] / math.pi) * (erec_meta["energy_mj"] / 1000.0) * i_corr

        p_main_chip = cond_result["p_cond_main"] + p_sw_main_chip
        p_diode_chip = cond_result["p_cond_diode"] + p_sw_diode_chip
        p_total_arm = (p_main_chip + p_diode_chip) * inputs["n_sim"]

        if "闭环" in inputs["sim_mode"]:
            tj_main_new = inputs["t_case_main"] + p_main_chip * inputs["rth_jc_main"]
            tj_diode_new = inputs["t_case_diode"] + p_diode_chip * inputs["rth_jc_diode"]
        else:
            tj_main_new, tj_diode_new = inputs["fixed_tj"], inputs["fixed_tj"]

        iteration_rows.append({"Iter": loop_idx + 1, "Tj_main_used (℃)": round(tj_main_current, 6), "Tj_diode_used (℃)": round(tj_diode_current, 6), "P_main_chip (W)": round(p_main_chip, 6), "P_diode_chip (W)": round(p_diode_chip, 6), "Tj_main_new (℃)": round(tj_main_new, 6), "Tj_diode_new (℃)": round(tj_diode_new, 6)})
        tj_main_current, tj_diode_current = tj_main_new, tj_diode_new

        if "闭环" in inputs["sim_mode"] and (loop_idx + 1) >= min_iter_before_break and max(abs(tj_main_new - iteration_rows[-1]["Tj_main_used (℃)"]), abs(tj_diode_new - iteration_rows[-1]["Tj_diode_used (℃)"])) < tolerance: break

    dominant_tj = max(tj_main_current, tj_diode_current)

    loss_breakdown_df = pd.DataFrame([
        {"对象": "主开关芯片", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_main"], "单臂总计 (W)": cond_result["p_cond_main"] * inputs["n_sim"]},
        {"对象": "主开关芯片", "项目": "开通+关断损耗", "单颗 (W)": p_sw_main_chip, "单臂总计 (W)": p_sw_main_chip * inputs["n_sim"]},
        {"对象": "续流二极管", "项目": "导通损耗", "单颗 (W)": cond_result["p_cond_diode"], "单臂总计 (W)": cond_result["p_cond_diode"] * inputs["n_sim"]},
        {"对象": "续流二极管", "项目": "恢复损耗", "单颗 (W)": p_sw_diode_chip, "单臂总计 (W)": p_sw_diode_chip * inputs["n_sim"]},
    ])

    linearized_df = pd.DataFrame([
        {"对象": "主开关芯片", "拟合区间": f"{main_model['i_low']:.1f}A ~ {main_model['i_high']:.1f}A", "V0 (V)": main_model["v0"], "R_eq (Ω)": main_model["r_eq"]},
        {"对象": "续流二极管", "拟合区间": f"{diode_model['i_low']:.1f}A ~ {diode_model['i_high']:.1f}A", "V0 (V)": diode_model["v0"], "R_eq (Ω)": diode_model["r_eq"]},
    ])

    return {
        "device_type": inputs["device_type"], "op_mode": inputs["op_mode"], "n_sim": inputs["n_sim"], "n_arm_system": inputs["n_arm_system"],
        "i_pk_chip": i_pk_chip, "dead_meta": dead_meta, "eon_meta": eon_meta, "eoff_meta": eoff_meta, "erec_meta": erec_meta,
        "p_main_chip": p_main_chip, "p_diode_chip": p_diode_chip, "p_total_arm": p_total_arm,
        "p_main_switch_position": p_main_chip * inputs["n_sim"], "p_diode_switch_position": p_diode_chip * inputs["n_sim"],
        "dominant_tj": dominant_tj, "tj_main_current": tj_main_current, "tj_diode_current": tj_diode_current,
        "loss_breakdown_df": loss_breakdown_df, "linearized_df": linearized_df,
    }

# ================= UI 渲染部分 =================
st.title("🛡️ 功率模块全工况电热联合仿真平台 (完全体物理对齐版)")

with st.expander("📝 工程随手记 & 快速操作指南", expanded=True):
    st.markdown("""
    **🚀 物理对齐核心特性：**
    1. **就近拟合 (V0/Req)**：不再傻瓜式地连接 0 点，而是自动寻找 V-I 表格中距离 $I_{pk}$ 最近的**两个物理点**进行真实斜率截距拟合。
    2. **温漂绝对控制**：不管表格里填了多少数据，只要你填了 **温漂系数**，底层都会坚定不移地乘上 `[1 + T_coeff * (Tj - Tref)]`。如果公司数据没有算温漂，请在此框**坚决填 0**。
    3. **双脉冲对齐**：支持切换“CAE相邻两点插值”或“公司最近单点等比例放缩”。
    """)

with st.sidebar:
    st.header("⚙️ 核心技术架构")
    device_type = st.radio("1. 模块芯片技术类型", ["IGBT + FRD (传统硅基)", "SiC MOSFET (碳化硅)"])
    st.divider()
    st.header("🧮 原始数据规格 (必填)")
    cond_data_type = st.radio("A. 导通 V-I 表格代表：", ["单芯片数据 (Bare Die)", "模块半桥数据 (Module)"])
    n_src_cond = st.number_input("V-I 原测模块芯片数", value=6, min_value=1, disabled="单芯片" in cond_data_type)
    sw_data_type = st.radio("B. 开关 E-I 表格代表：", ["模块半桥数据 (Module)", "单芯片数据 (Bare Die)"])
    n_src_sw = st.number_input("E-I 原测模块芯片数", value=6, min_value=1, disabled="单芯片" in sw_data_type)
    st.divider()
    st.header("🎯 仿真目标规模重构")
    n_sim = st.number_input("目标仿真单臂芯片数 (N_sim)", value=6, min_value=1)
    n_arm_system = st.number_input("系统桥臂数 (N_arm_sys)", value=1, min_value=1)
    st.divider()
    st.header("🔄 热学计算工作流")
    sim_mode = st.radio("模式选择", ["A. 开环盲算 (已知结温)", "B. 闭环迭代 (已知热阻)"])
    if "闭环" in sim_mode:
        split_thermal_params = st.checkbox("主/二极管分开热参数", value=True)
        rth_jc_main = st.number_input("主芯片 RthJC_main (K/W)", value=0.065, format="%.4f")
        t_case_main = st.number_input("主芯片 Tc_main (℃)", value=65.0)
        rth_jc_diode = st.number_input("二极管 RthJC_diode (K/W)", value=0.085, format="%.4f") if split_thermal_params else rth_jc_main
        t_case_diode = st.number_input("二极管 Tc_diode (℃)", value=65.0) if split_thermal_params else t_case_main
        fixed_tj = None
    else:
        fixed_tj = st.number_input("设定全局目标结温 Tj (℃)", value=150.0)
        split_thermal_params = False; rth_jc_main = None; rth_jc_diode = None; t_case_main = None; t_case_diode = None

st.divider()
st.header("📊 第一步：特性数据录入 (归一化中心)")
col_main, col_diode = st.columns(2)
with col_main:
    st.subheader("🔴 主开关管 (IGBT / SiC)")
    ev_main = st.data_editor(DEFAULT_MAIN_VI, num_rows="dynamic", key="v_main")
    ee_main = st.data_editor(DEFAULT_MAIN_EI, num_rows="dynamic", key="e_main")
with col_diode:
    st.subheader("🔵 续流二极管 (FRD / Body Diode)")
    ev_diode = st.data_editor(DEFAULT_DIODE_VI, num_rows="dynamic", key="v_diode")
    ee_diode = st.data_editor(DEFAULT_DIODE_EI, num_rows="dynamic", key="ee_diode")

st.divider()
st.header("⚙️ 第二步：全场景工况与物理修正系数")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**⚡ 车辆 / 电驱动工况**")
    op_mode = st.selectbox("🏎️ 运行场景切换", ["电动/巡航 (Motoring)", "制动/反拖 (Regeneration)", "最恶劣堵转 (Stall)"])
    vdc_act = st.number_input("母线 V_dc (V)", value=713.0, min_value=0.0)
    iout_rms = st.number_input("有效值 I_out (A)", value=264.5, min_value=0.0)
    fsw = st.number_input("开关频率 f_sw (Hz)", value=10000.0, min_value=0.0)
    fout = st.number_input("输出频率 f_out (Hz)", value=200.0, min_value=0.0)
    m_index = st.number_input("调制系数 M", value=0.90, min_value=0.0, max_value=1.15)
    cosphi = st.number_input("功率因数幅值 cos_phi", value=0.90, min_value=0.0, max_value=1.0)
    mode = st.selectbox("调制模式选择", ["SVPWM", "SPWM"])

with c2:
    st.markdown("**🔬 核心对标策略 (重点)**")
    algo_type = st.radio("开关能量提取算法", ["1. CAE相邻两点曲线插值", "2. 最近单点等比例法(对标公司)"], help="比例法会自动寻找离输出电流最近的表格测试点，然后直线放缩。")
    v_ref = st.number_input("规格书基准 V_nom (V)", value=600.0, min_value=0.001)
    t_ref_dp = st.number_input("规格书基准 T_ref (℃)", value=150.0)

with c3:
    st.markdown("**📈 拟合修正指数**")
    kv_on = st.number_input("开通指数 K_v_on", value=1.30)
    kv_off = st.number_input("关断指数 K_v_off", value=1.30)
    kv_frd = st.number_input("续流指数 K_v_frd", value=0.60)
    ki_frd = st.number_input("续流电流指数 K_i_frd", value=0.60)
    kron = st.number_input("电阻系数 K_ron", value=0.30)
    kroff = st.number_input("关断电阻系数 K_roff (坚守项)", value=0.50)
    dead_time_us = st.number_input("死区时间 t_dead (us)", value=2.0, min_value=0.0)

with c4:
    st.markdown("**🌡️ 温漂 / 驱动 / 寄生电阻**")
    st.info("若公司报告未计算温漂，请务必将下方填为 0.0000")
    t_coeff_igbt = st.number_input("主管温漂系数 (1/K)", value=0.003, format="%.4f")
    t_coeff_frd = st.number_input("续流温漂系数 (1/K)", value=0.006 if "IGBT" in device_type else 0.003, format="%.4f")
    rg_on_ref = st.number_input("手册 R_g,on (Ω)", value=2.5, min_value=0.0)
    rg_off_ref = st.number_input("手册 R_g,off (Ω)", value=20.0, min_value=0.0)
    rg_on_act = st.number_input("实际 R_on (Ω)", value=2.5, min_value=0.0)
    rg_off_act = st.number_input("实际 R_off (Ω)", value=20.0, min_value=0.0)
    r_pkg_mohm = st.number_input("封装内阻 R_pkg,chip (mΩ)", value=0.0, min_value=0.0)
    r_arm_mohm = st.number_input("桥臂附加电阻 R_arm (mΩ)", value=0.0, min_value=0.0)

inputs = {
    "device_type": device_type, "cond_data_type": cond_data_type, "n_src_cond": int(n_src_cond),
    "sw_data_type": sw_data_type, "n_src_sw": int(n_src_sw), "n_sim": int(n_sim), "n_arm_system": int(n_arm_system),
    "sim_mode": sim_mode, "split_thermal_params": bool(split_thermal_params), "rth_jc_main": 0.0 if rth_jc_main is None else float(rth_jc_main), "rth_jc_diode": 0.0 if rth_jc_diode is None else float(rth_jc_diode),
    "t_case_main": 0.0 if t_case_main is None else float(t_case_main), "t_case_diode": 0.0 if t_case_diode is None else float(t_case_diode), "fixed_tj": 150.0 if fixed_tj is None else float(fixed_tj),
    "op_mode": op_mode, "vdc_act": float(vdc_act), "iout_rms": float(iout_rms), "fsw": float(fsw), "fout": float(fout), "m_index": float(m_index), "cosphi": float(cosphi), "mode": mode,
    "v_ref": float(v_ref), "t_ref_dp": float(t_ref_dp), "rg_on_ref": float(rg_on_ref), "rg_off_ref": float(rg_off_ref), "rg_on_act": float(rg_on_act), "rg_off_act": float(rg_off_act),
    "algo_type": algo_type, "dead_time_us": float(dead_time_us),
    "kv_on": float(kv_on), "kv_off": float(kv_off), "kv_frd": float(kv_frd), "ki_frd": float(ki_frd), "kron": float(kron), "kroff": float(kroff),
    "t_coeff_igbt": float(t_coeff_igbt), "t_coeff_frd": float(t_coeff_frd), "r_pkg_mohm": float(r_pkg_mohm), "r_arm_mohm": float(r_arm_mohm),
}

tables_for_validation = {"ev_main": validate_numeric_table(ev_main, "主开关管导通表", [TEMP_COL, CURRENT_COL, "V_drop (V)"])[0],
                         "ee_main": validate_numeric_table(ee_main, "主开关管开关能量表", [TEMP_COL, CURRENT_COL, "Eon (mJ)", "Eoff (mJ)"])[0],
                         "ev_diode": validate_numeric_table(ev_diode, "二极管导通表", [TEMP_COL, CURRENT_COL, "Vf (V)"])[0],
                         "ee_diode": validate_numeric_table(ee_diode, "二极管恢复能量表", [TEMP_COL, CURRENT_COL, "Erec (mJ)"])[0]}

st.divider()
st.header("🚀 第三步：执行全工况联合仿真")

compute_requested = st.button("🚀 执 行 全 工 况 仿 真 计 算", use_container_width=True)

if compute_requested:
    result = simulate_system(inputs, tables_for_validation)
    st.session_state["simulation_result"] = result

result = st.session_state.get("simulation_result")

if result:
    st.success(f"✅ 计算完成！数据已按最近工作点进行提取与物理补偿。")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("控制结温 Tj,max", f"{result['dominant_tj']:.1f} ℃")
    m2.metric("主芯片结温", f"{result['tj_main_current']:.1f} ℃")
    m3.metric("二极管结温", f"{result['tj_diode_current']:.1f} ℃")
    m4.metric("单臂总功耗", f"{result['p_total_arm']:.1f} W")
    m5.metric("系统级总功耗", f"{result['p_total_arm'] * result['n_arm_system']:.1f} W")

    st.divider()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("🔴 主芯片单颗发热率 (填入CAE)", f"{result['p_main_chip']:.2f} W")
    p2.metric("🔵 二极管单颗发热率 (填入CAE)", f"{result['p_diode_chip']:.2f} W")
    p3.metric("🔴 主开关并联位总损耗 (对标报告)", f"{result['p_main_switch_position']:.1f} W")
    p4.metric("🔵 续流并联位总损耗 (对标报告)", f"{result['p_diode_switch_position']:.1f} W")

    st.info(f"🔍 追溯分析：本次计算开关管 E-I 采用 **{result['eon_meta']['extraction_label']}**。主开关 V0/Req 在 **{result['linearized_df'].iloc[0]['拟合区间']}** 内拉切线得到。")

    tabs = st.tabs(["结果总览", "动态线性化模型跟踪"])
    with tabs[0]:
        st.dataframe(result["loss_breakdown_df"], use_container_width=True)
    with tabs[1]:
        st.markdown("**证明：V0 和 Req 是根据 $I_{pk}$ 所在的电流区间动态截取的真实切线，而非强行拉零：**")
        st.dataframe(result["linearized_df"], use_container_width=True)
