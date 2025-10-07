
# dashboard_pro_v2.py
# Enhanced Power Distribution Digital Twin Dashboard
# Adds: Alert system, Trends, Forecasting (graceful fallback), Control simulation, Suggestions, System Overview,
# Exporting, Alerts log, UI improvements (logo upload, dark theme toggle).
#
# NOTE: This file is intended to run with Streamlit. This generator created the file and performed a syntax check.
# To run locally: `streamlit run dashboard_pro_v2.py` (ensure your CSV files are in the same folder or update paths).
# If `statsmodels` is not installed, forecasting falls back to a simple trend extrapolation.

import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import os
import io
import json
import plotly.graph_objs as go

# Optional forecasting library (used if available)
try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

# ----------------------------------------------------------
# CONFIG / PATHS (update paths if needed)
# ----------------------------------------------------------
st.set_page_config(page_title="Power Distribution DT - Pro v2", layout="wide")

CSV_PATH = "Total_P&Q.csv"
FEEDER_A_P = "FeederA_P.csv"
FEEDER_A_Q = "FeederA_Q.csv"
FEEDER_A_V = "FeederA_Bus_pu_voltages.csv"
FEEDER_B_P = "FeederB_P.csv"
FEEDER_B_Q = "FeederB_Q.csv"
FEEDER_B_V = "FeederB_Bus_pu_voltages.csv"
FEEDER_C_P = "FeederC_P.csv"
FEEDER_C_Q = "FeederC_Q.csv"
FEEDER_C_V = "FeederC_Bus_pu_voltages.csv"

ALERT_LOG_KEY = "alerts"  # session_state key for alerts

# ----------------------------------------------------------
# HELPERS & PLOTTING (kept deliberately simple & robust)
# ----------------------------------------------------------
@st.cache_data
def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # try with different encodings or separators if required
        return pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

def make_line_fig(x, y, title, yaxis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title,
                             hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(margin=dict(l=20,r=20,t=32,b=20),
                      title=dict(text=title, x=0.01, xanchor='left'),
                      yaxis=dict(title=yaxis_title),
                      xaxis=dict(title="Time"))
    return fig

def make_gauge(value, title, unit):
    axis_max = max(1.0, abs(value) * 1.5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': f"<b>{title}</b>", 'font': {'size': 16}},
        number={'suffix': f" {unit}", 'font': {'size': 22}},
        delta={'reference': 0, 'relative': False},
        gauge={ 'axis': {'range': [0, axis_max], 'tickwidth': 1},
                'bar': {'color': "#4CAF50"},
                'bgcolor': "white",
                'borderwidth': 1, 'bordercolor': "gray"}
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def make_dual_line_fig(x, y1, y2, title, y1_title, y2_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines", name=y1_title, hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name=y2_title, hovertemplate="%{x}<br>%{y:.2f}<extra></extra>", yaxis="y2"))
    fig.update_layout(margin=dict(l=20,r=20,t=32,b=20),
                      title=dict(text=title, x=0.01, xanchor='left'),
                      yaxis=dict(title=y1_title),
                      yaxis2=dict(title=y2_title, overlaying="y", side="right"),
                      xaxis=dict(title="Time"))
    return fig

def make_voltage_fig(x, voltages_df, v_cols, tap_cols, bus_name):
    fig = go.Figure()
    phase_colors = ["red","green","blue"]
    for i, col in enumerate(v_cols):
        if col in voltages_df.columns:
            fig.add_trace(go.Scatter(x=x, y=voltages_df[col], mode="lines", name=f"Voltage Phase {i+1}", hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
    tap_colors = ["orange","purple","brown"]
    for i, col in enumerate(tap_cols):
        if col in voltages_df.columns:
            fig.add_trace(go.Scatter(x=x, y=voltages_df[col], mode="lines", name=f"Tap Changer {i+1}", hovertemplate="%{x}<br>%{y:.4f}<extra></extra>", line=dict(dash="dash")))
    fig.update_layout(margin=dict(l=20,r=20,t=32,b=20),
                      title=dict(text=f"{bus_name} - Bus Voltages (PU) & Tap Changers", x=0.01, xanchor='left'),
                      yaxis=dict(title="PU"), xaxis=dict(title="Time"))
    return fig

def compute_power_factor(P, Q):
    S = np.sqrt(np.square(P) + np.square(Q))
    # avoid division by zero
    pf = np.where(S==0, 0.0, np.abs(P) / S)
    return pf

def add_alert(message, severity="warning"):
    # severity: "info", "warning", "error"
    now = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    alert = {"time": now, "msg": message, "severity": severity}
    alerts = st.session_state.get(ALERT_LOG_KEY, [])
    # avoid repeating the same alert frequently - simple duplicate check
    if not alerts or alerts[-1]["msg"] != message:
        alerts.append(alert)
        # keep only last 200 alerts
        st.session_state[ALERT_LOG_KEY] = alerts[-200:]

def get_alerts():
    return st.session_state.get(ALERT_LOG_KEY, [])

def clear_alerts():
    st.session_state[ALERT_LOG_KEY] = []

# Forecast helper (graceful fallback)
def forecast_series(series, steps=24):
    # series: pandas Series indexed by datetime
    series = series.dropna()
    if series.empty:
        return pd.Series([0.0]*steps, index=pd.date_range(start=pd.Timestamp.now(), periods=steps, freq='h'))
    try:
        if HAS_ARIMA and len(series) > 30:
            model = ARIMA(series, order=(2,1,2))
            res = model.fit()
            f = res.forecast(steps=steps)
            idx = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=steps, freq='h')
            f.index = idx
            return f
    except Exception as e:
        # fallback below
        pass
    # simple linear trend extrapolation using last 48 points
    window = min(len(series), 48)
    y = series.values[-window:]
    x = np.arange(window)
    # least squares poly fit degree 1 (linear)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    future_x = np.arange(window, window + steps)
    pred = intercept + slope * future_x
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=steps, freq='h')
    return pd.Series(pred, index=idx)

# Apply control simulation to feed-level dataframe copies (returns modified copy)
def apply_controls_to_df(df_p, df_q, control_state):
    df_p2 = df_p.copy()
    df_q2 = df_q.copy()
    # Apply load shedding to selected feeder(s)
    shed_pct = control_state.get("shed_pct", 0.0) / 100.0
    if shed_pct > 0.0 and control_state.get("shed_target", "All") in ("All", "This Feeder"):
        df_p2 = df_p2 * (1.0 - shed_pct)
        df_q2 = df_q2 * (1.0 - shed_pct)
    # Apply capacitor reactive compensation (reduces Q)
    if control_state.get("cap_on", False):
        cap_eff = control_state.get("cap_effect_pct", 10.0) / 100.0
        df_q2 = df_q2 * (1.0 - cap_eff)
    # Tap adjustments could be simulated by slightly shifting voltages outside of this function (handled elsewhere)
    return df_p2, df_q2

# ----------------------------------------------------------
# LOAD DATA (cached)
# ----------------------------------------------------------
@st.cache_data
def load_all():
    out = {}
    out['total'] = safe_read_csv(CSV_PATH)
    out['A_p'] = safe_read_csv(FEEDER_A_P)
    out['A_q'] = safe_read_csv(FEEDER_A_Q)
    out['A_v'] = safe_read_csv(FEEDER_A_V)
    out['B_p'] = safe_read_csv(FEEDER_B_P)
    out['B_q'] = safe_read_csv(FEEDER_B_Q)
    out['B_v'] = safe_read_csv(FEEDER_B_V)
    out['C_p'] = safe_read_csv(FEEDER_C_P)
    out['C_q'] = safe_read_csv(FEEDER_C_Q)
    out['C_v'] = safe_read_csv(FEEDER_C_V)
    return out

data_store = load_all()

# ----------------------------------------------------------
# SIDEBAR - Global Controls, Simulation settings, thresholds
# ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Power Distribution DT - Controls")
    nav = st.radio("Navigation", ["Home", "Feeder A", "Feeder B", "Feeder C", "System Overview", "Trends", "Control Center", "Alerts", "Settings"])
    st.markdown("---")
    st.subheader("Simulation Settings")
    start_date = st.date_input("Start date (used if no timestamp)", value=datetime.date(2024,1,1))
    update_interval = st.slider("Update interval (seconds)", 1, 10, 2)
    window_size = st.slider("Chart window (points shown)", 20, 2000, 200)
    loop_dataset = st.checkbox("Loop dataset", value=False)
    st.markdown("---")
    st.subheader("Alert Thresholds (editable)")
    voltage_low = st.number_input("Voltage low (PU)", value=0.95, step=0.01, format="%.3f")
    voltage_high = st.number_input("Voltage high (PU)", value=1.05, step=0.01, format="%.3f")
    feeder_limit_default = 1000.0
    feeder_limit = st.number_input("Feeder overload limit (kW)", value=feeder_limit_default, step=100.0)
    reactive_bad_pct = st.number_input("Reactive threshold (kVAR)", value=500.0, step=50.0)
    st.markdown("---")
    st.subheader("Simulation Buttons")
    start_btn = st.button("▶ Start")
    stop_btn = st.button("⏸ Stop")
    reset_btn = st.button("↺ Reset")
    st.markdown("---")
    st.subheader("Logo / Theme")
    logo_file = st.file_uploader("Upload logo (optional)", type=['png','jpg','jpeg'])
    dark_mode = st.checkbox("Dark theme (UI only)", value=False)

# Handle logo upload (save to assets/ if provided)
if 'logo_path' not in st.session_state:
    st.session_state.logo_path = None

if logo_file is not None:
    logo_bytes = logo_file.getvalue()
    os.makedirs("assets", exist_ok=True)
    logo_path = os.path.join("assets", "company_logo.png")
    with open(logo_path, "wb") as f:
        f.write(logo_bytes)
    st.session_state.logo_path = logo_path

# ----------------------------------------------------------
# SESSION STATE defaults (safe initializations)
# ----------------------------------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'start_date' not in st.session_state:
    st.session_state.start_date = start_date
if ALERT_LOG_KEY not in st.session_state:
    st.session_state[ALERT_LOG_KEY] = []

# simulation control state used in control center
if 'control_state' not in st.session_state:
    st.session_state.control_state = {"cap_on": False, "cap_effect_pct": 10.0, "shed_pct": 0.0, "shed_target": "All"}

# control buttons behavior
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.running = False
    st.session_state.idx = 0

# update start date change resets index
if start_date != st.session_state.start_date:
    st.session_state.start_date = start_date
    st.session_state.idx = 0

# show logo if available
if st.session_state.logo_path and os.path.exists(st.session_state.logo_path):
    st.sidebar.image(st.session_state.logo_path, use_column_width=True)

# dark mode (simple background/text switch via CSS)
if dark_mode:
    st.markdown("""
    <style>
    .main { background-color: #0f1720; color: #cbd5e1; }
    .stApp { background-color: #0f1720; color: #cbd5e1; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# NAV: HOME
# ----------------------------------------------------------
if nav == "Home":
    st.header("🏠 Home — Real-Time Power Flow Dashboard (Pro)")
    st.write("Simulating smooth, real-time Total Active (kW) and Reactive (kVAR) power. Alerts & suggestions appear below.")

    df_raw = data_store['total'] if not data_store['total'].empty else pd.DataFrame(columns=["Total_Active_Power","Total_Reac_Power"])
    n = len(df_raw)
    # create index if missing
    if "timestamp" not in df_raw.columns:
        idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
        df_raw.index = idx
    else:
        try:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            df_raw = df_raw.set_index('timestamp')
        except Exception:
            idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
            df_raw.index = idx

    # placeholders and layout
    top = st.container()
    with top:
        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown("### ⚡ Real-Time Total Power Monitoring")
            st.caption("Streaming one timestamp at a time for simulated live visualization.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    gauge_col1, gauge_col2 = st.columns(2)
    metric_col1, metric_col2 = st.columns(2)
    chart_col1, chart_col2 = st.columns(2)
    alert_col, suggest_col = st.columns([2,3])

    gauge_ph_1 = gauge_col1.empty()
    gauge_ph_2 = gauge_col2.empty()
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()
    chart_ph_1 = chart_col1.empty()
    chart_ph_2 = chart_col2.empty()
    alert_ph = alert_col.empty()
    suggest_ph = suggest_col.empty()
    status_ph = st.empty()

    # initial draw
    current_idx = max(0, min(st.session_state.idx, n-1))
    history = df_raw.iloc[:current_idx+1] if current_idx>=0 else pd.DataFrame()

    if history.empty:
        chart_ph_1.plotly_chart(make_line_fig([], [], "Total Active Power (kW)", "kW"), use_container_width=True)
        chart_ph_2.plotly_chart(make_line_fig([], [], "Total Reactive Power (kVAR)", "kVAR"), use_container_width=True)
        gauge_ph_1.plotly_chart(make_gauge(0, "Active Power", "kW"), use_container_width=True)
        gauge_ph_2.plotly_chart(make_gauge(0, "Reactive Power", "kVAR"), use_container_width=True)

    # streaming loop (keeps old while structure; careful to exit by changing nav or pressing stop)
    try:
        while st.session_state.running and nav == "Home":
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success("✅ Reached end of dataset. Press Reset or enable Loop to restart.")
                    break

            current_idx = max(0, min(st.session_state.idx, n-1))
            history = df_raw.iloc[:current_idx+1]

            if history.empty:
                active_val = 0.0
                reactive_val = 0.0
                delta_active = 0.0
                delta_reactive = 0.0
            else:
                active_val = history["Total_Active_Power"].iloc[-1] if "Total_Active_Power" in history.columns else 0.0
                reactive_val = history["Total_Reac_Power"].iloc[-1] if "Total_Reac_Power" in history.columns else 0.0
                if len(history) >= 2:
                    delta_active = active_val - (history["Total_Active_Power"].iloc[-2] if "Total_Active_Power" in history.columns else 0.0)
                    delta_reactive = reactive_val - (history["Total_Reac_Power"].iloc[-2] if "Total_Reac_Power" in history.columns else 0.0)
                else:
                    delta_active = 0.0
                    delta_reactive = 0.0

            # metrics & gauges
            metric_ph_1.metric("Total Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric("Total Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")
            gauge_ph_1.plotly_chart(make_gauge(active_val, "Active Power", "kW"), use_container_width=True)
            gauge_ph_2.plotly_chart(make_gauge(reactive_val, "Reactive Power", "kVAR"), use_container_width=True)

            # update line charts
            start_window = max(0, len(history)-window_size)
            hist_win = history.iloc[start_window:]
            fig_active = make_line_fig(hist_win.index, hist_win["Total_Active_Power"] if "Total_Active_Power" in hist_win.columns else [], "Total Active Power Over Time", "Active Power (kW)")
            fig_reactive = make_line_fig(hist_win.index, hist_win["Total_Reac_Power"] if "Total_Reac_Power" in hist_win.columns else [], "Total Reactive Power Over Time", "Reactive Power (kVAR)")
            chart_ph_1.plotly_chart(fig_active, use_container_width=True)
            chart_ph_2.plotly_chart(fig_reactive, use_container_width=True)

            # Alert checks (simple rule-based)
            if "Total_Active_Power" in history.columns and active_val > feeder_limit:
                add_alert(f"Overload: Total active power {active_val:.2f} kW exceeded feeder limit ({feeder_limit} kW).", "error")
            if "Total_Reac_Power" in history.columns and abs(reactive_val) > reactive_bad_pct:
                add_alert(f"High reactive power: {reactive_val:.2f} kVAR (threshold {reactive_bad_pct}).", "warning")

            # show recent alerts
            alerts = get_alerts()
            if alerts:
                with alert_ph.container():
                    st.markdown("### 🔔 Recent Alerts")
                    for a in alerts[-6:][::-1]:
                        if a["severity"] == "error":
                            st.error(f"{a['time']}  —  {a['msg']}")
                        elif a["severity"] == "warning":
                            st.warning(f"{a['time']}  —  {a['msg']}")
                        else:
                            st.info(f"{a['time']}  —  {a['msg']}")
            else:
                alert_ph.info("No alerts (yet).")

            # Smart suggestions: simple heuristics
            suggestions = []
            if "Total_Active_Power" in history.columns and active_val > 0.9 * feeder_limit:
                suggestions.append(f"Feeder approaching limit ({active_val:.2f} kW). Consider shedding 5-15% load.")
            if "Total_Reac_Power" in history.columns and abs(reactive_val) > reactive_bad_pct:
                suggestions.append("Reactive power high — suggest enabling capacitor bank or reactive compensation.")
            if suggestions:
                with suggest_ph.container():
                    st.markdown("### 💡 Suggestions")
                    for s in suggestions:
                        st.info(s)
            else:
                suggest_ph.info("No suggestions at the moment. System stable.")

            # iterate
            st.session_state.idx += 1
            status_ph.info(f"Streaming Home... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            st.rerun()
    except Exception as e:
        status_ph.error(f"Home loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info("Simulation paused for Home. Press ▶ Start to resume or ↺ Reset.")

# ----------------------------------------------------------
# NAV: FEEDER TEMPLATES (A / B / C)
# ----------------------------------------------------------
elif nav in ("Feeder A", "Feeder B", "Feeder C"):
    feeder_letter = nav.split()[-1]  # "A" / "B" / "C"
    st.header(f"🔌 Feeder {feeder_letter} — Bus-Level Monitoring & Controls")
    st.write("Active (kW), Reactive (kVAR), and Voltage (PU) updates. Use Control Center to simulate actions.")

    # select data depending on feeder
    if feeder_letter == "A":
        df_p = data_store['A_p'].copy()
        df_q = data_store['A_q'].copy()
        df_v = data_store['A_v'].copy()
    elif feeder_letter == "B":
        df_p = data_store['B_p'].copy()
        df_q = data_store['B_q'].copy()
        df_v = data_store['B_v'].copy()
    else:
        df_p = data_store['C_p'].copy()
        df_q = data_store['C_q'].copy()
        df_v = data_store['C_v'].copy()

    # ensure there's at least empty frames with columns
    if df_p.empty:
        st.warning(f"No active power CSV found for Feeder {feeder_letter}.")
        df_p = pd.DataFrame()
    if df_q.empty:
        st.warning(f"No reactive power CSV found for Feeder {feeder_letter}.")
        df_q = pd.DataFrame()
    if df_v.empty:
        st.info(f"No voltage CSV found for Feeder {feeder_letter}. Voltage plots will be disabled.")

    buses = df_p.columns.tolist() if not df_p.empty else []
    if 'selected_bus' not in st.session_state or st.session_state.get('selected_bus') not in buses:
        st.session_state.selected_bus = buses[0] if buses else None
    selected_bus = st.selectbox("Select Bus", options=buses, index=buses.index(st.session_state.selected_bus) if st.session_state.selected_bus in buses else 0)
    st.session_state.selected_bus = selected_bus

    # ensure timestamp index
    n = len(df_p)
    if n > 0:
        idx = pd.date_range(start=st.session_state.start_date, periods=n, freq="h")
        df_p.index = idx
        df_q.index = idx
        if not df_v.empty:
            df_v.index = idx

    # placeholders
    top = st.container()
    with top:
        c1, c2 = st.columns([3,1])
        with c1:
            st.markdown(f"### 📊 {selected_bus} Power & Voltage Monitoring")
            st.caption("Streaming simulated real-time data.")
        with c2:
            st.metric("Total Data Points", f"{n:,}")

    metric_col1, metric_col2 = st.columns(2)
    metric_ph_1 = metric_col1.empty()
    metric_ph_2 = metric_col2.empty()
    power_ph = st.empty()
    voltage_ph = st.empty()
    status_ph = st.empty()

    # initial plots
    power_ph.plotly_chart(make_dual_line_fig([], [], [], f"{selected_bus} Active & Reactive Power", "kW", "kVAR"), use_container_width=True)
    voltage_ph.plotly_chart(make_voltage_fig([], pd.DataFrame(), [], [], selected_bus), use_container_width=True)

    # live feed loop
    try:
        while st.session_state.running and nav == f"Feeder {feeder_letter}":
            if st.session_state.idx >= n:
                if loop_dataset:
                    st.session_state.idx = 0
                else:
                    st.session_state.running = False
                    status_ph.success(f"✅ Reached end of Feeder {feeder_letter} dataset.")
                    break

            current_idx = max(0, min(st.session_state.idx, n-1))
            history_p = df_p.iloc[:current_idx+1]
            history_q = df_q.iloc[:current_idx+1]
            history_v = df_v.iloc[:current_idx+1] if not df_v.empty else pd.DataFrame()

            if selected_bus not in df_p.columns or selected_bus not in df_q.columns:
                status_ph.error(f"Selected bus '{selected_bus}' not found in Feeder {feeder_letter} data.")
                break

            # apply control simulation overlays (non-destructive)
            sim_p, sim_q = apply_controls_to_df(history_p, history_q, st.session_state.control_state)

            active_val = sim_p[selected_bus].iloc[-1] if not sim_p.empty else 0.0
            reactive_val = sim_q[selected_bus].iloc[-1] if not sim_q.empty else 0.0

            if len(sim_p) >= 2:
                delta_active = active_val - sim_p[selected_bus].iloc[-2]
            else:
                delta_active = 0.0
            if len(sim_q) >= 2:
                delta_reactive = reactive_val - sim_q[selected_bus].iloc[-2]
            else:
                delta_reactive = 0.0

            metric_ph_1.metric(f"{selected_bus} Active Power (kW)", f"{active_val:,.2f}", f"{delta_active:+.2f}")
            metric_ph_2.metric(f"{selected_bus} Reactive Power (kVAR)", f"{reactive_val:,.2f}", f"{delta_reactive:+.2f}")

            # update power graphs
            start_window = max(0, len(sim_p) - window_size)
            hist_p_win = sim_p.iloc[start_window:]
            hist_q_win = sim_q.iloc[start_window:]
            fig_power = make_dual_line_fig(hist_p_win.index, hist_p_win[selected_bus] if not hist_p_win.empty else [], hist_q_win[selected_bus] if not hist_q_win.empty else [], f"{selected_bus} Active & Reactive Power Over Time", "Active Power (kW)", "Reactive Power (kVAR)")
            power_ph.plotly_chart(fig_power, use_container_width=True)

            # voltage graph
            if not history_v.empty:
                # expect columns like 'busX.1', 'busX.2' etc. If not available, show warning
                bus_num = ''.join(filter(str.isdigit, str(selected_bus)))
                bus_lower = f"bus{bus_num}"
                v_phases = [f"{bus_lower}.{i}" for i in ['1','2','3']]
                t_taps = [f"t_{bus_lower}_l.{i}" for i in ['1','2','3']]
                missing_v = [col for col in v_phases + t_taps if col not in history_v.columns]
                if missing_v:
                    voltage_ph.warning(f"Voltage/tap columns missing for {selected_bus}: {missing_v}.")
                else:
                    hist_v_win = history_v.iloc[start_window:]
                    fig_voltage = make_voltage_fig(hist_v_win.index, hist_v_win, v_phases, t_taps, selected_bus)
                    voltage_ph.plotly_chart(fig_voltage, use_container_width=True)
            else:
                voltage_ph.info("No voltage data available for this feeder.")

            # quick alerts for bus-level issues
            if not history_v.empty:
                # last voltages (take mean of phases if present)
                vcols = [c for c in history_v.columns if str(selected_bus).lower().replace(' ','') in c or f"bus{''.join(filter(str.isdigit, str(selected_bus)))}" in c]
                # fallback: check any voltage below threshold in history_v
                recent_v = history_v.iloc[-1] if not history_v.empty else None
                if recent_v is not None:
                    any_low = any((recent_v < voltage_low) & ~recent_v.isna())
                    any_high = any((recent_v > voltage_high) & ~recent_v.isna())
                    if any_low:
                        add_alert(f"{selected_bus}: Voltage below {voltage_low} PU detected.", "warning")
                    if any_high:
                        add_alert(f"{selected_bus}: Voltage above {voltage_high} PU detected.", "warning")

            st.session_state.idx += 1
            status_ph.info(f"Streaming Feeder {feeder_letter}... Row {min(st.session_state.idx, n)}/{n}  •  Next update in {update_interval}s")
            time.sleep(update_interval)
            st.rerun()
    except Exception as e:
        status_ph.error(f"Feeder {feeder_letter} loop stopped: {e}")
    finally:
        if not st.session_state.running:
            status_ph.info(f"Simulation paused for Feeder {feeder_letter}. Press ▶ Start to resume or ↺ Reset.")

# ----------------------------------------------------------
# NAV: SYSTEM OVERVIEW
# ----------------------------------------------------------
elif nav == "System Overview":
    st.header("🛰️ System Overview — Feeder Comparison & Top Alerts")
    # Build summary stats from feeder totals (sum across all buses)
    feeders = []
    for letter, p_key, q_key in [("A","A_p","A_q"), ("B","B_p","B_q"), ("C","C_p","C_q")]:
        dfp = data_store[p_key]
        dfq = data_store[q_key]
        if not dfp.empty and not dfq.empty:
            total_p = dfp.sum(axis=1)
            total_q = dfq.sum(axis=1)
            totals = pd.DataFrame({"P": total_p, "Q": total_q})
            totals.index = pd.date_range(start=st.session_state.start_date, periods=len(totals), freq="h") if totals.index.empty else totals.index
            feeders.append((letter, totals))
    if feeders:
        # combine latest totals per feeder
        latest = {f"Feeder {f}": tot.iloc[-1]["P"] if not tot.empty else 0.0 for f, tot in feeders}
        names = list(latest.keys())
        values = list(latest.values())
        fig = go.Figure([go.Bar(x=names, y=values)])
        fig.update_layout(title="Latest Total Active Power by Feeder (kW)", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)
        # show table
        df_table = pd.DataFrame.from_dict(latest, orient='index', columns=["Active Power (kW)"])
        st.dataframe(df_table)
    else:
        st.info("No feeder summary data available. Make sure feeder CSVs are present.")

    # Top alerts
    st.markdown("---")
    st.subheader("Recent System Alerts")
    alerts = get_alerts()
    if alerts:
        for a in alerts[::-1][:20]:
            if a["severity"] == "error":
                st.error(f"{a['time']}  —  {a['msg']}")
            elif a["severity"] == "warning":
                st.warning(f"{a['time']}  —  {a['msg']}")
            else:
                st.info(f"{a['time']}  —  {a['msg']}")
        st.download_button("Download Alerts (JSON)", json.dumps(alerts, indent=2), "alerts.json")
    else:
        st.info("No alerts logged.")

# ----------------------------------------------------------
# NAV: TRENDS
# ----------------------------------------------------------
elif nav == "Trends":
    st.header("📈 Trends & Analytics")
    df_total = data_store['total'] if not data_store['total'].empty else pd.DataFrame(columns=["Total_Active_Power","Total_Reac_Power"])

    if df_total.empty:
        st.info("No total dataset available for trend analysis.")
    else:
        # ensure datetime index
        if "timestamp" not in df_total.columns:
            df_total.index = pd.date_range(start=st.session_state.start_date, periods=len(df_total), freq="h")
        else:
            try:
                df_total['timestamp'] = pd.to_datetime(df_total['timestamp'])
                df_total.set_index('timestamp', inplace=True)
            except Exception:
                df_total.index = pd.date_range(start=st.session_state.start_date, periods=len(df_total), freq="h")
        # compute rolling metrics
        df_total['P_24h_avg'] = df_total["Total_Active_Power"].rolling(window=24, min_periods=1).mean()
        df_total['Q_24h_avg'] = df_total["Total_Reac_Power"].rolling(window=24, min_periods=1).mean()
        df_total['pf'] = compute_power_factor(df_total["Total_Active_Power"].fillna(0.0), df_total["Total_Reac_Power"].fillna(0.0))

        tab1, tab2 = st.tabs(["Smoothed Trends", "Forecast"])
        with tab1:
            st.subheader("24-hour Rolling Averages")
            st.plotly_chart(make_line_fig(df_total.index, df_total['P_24h_avg'], "24h Avg Active Power (kW)", "kW"), use_container_width=True)
            st.plotly_chart(make_line_fig(df_total.index, df_total['Q_24h_avg'], "24h Avg Reactive Power (kVAR)", "kVAR"), use_container_width=True)
            st.plotly_chart(make_line_fig(df_total.index, df_total['pf'], "Estimated System Power Factor (pf)", "PF"), use_container_width=True)

        with tab2:
            st.subheader("Forecast Next 24 Hours (Total Active Power)")
            series = df_total["Total_Active_Power"].dropna()
            forecast = forecast_series(series, steps=24)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index[-200:], y=series.values[-200:], mode="lines", name="Historical"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(dash="dot")))
            st.plotly_chart(fig, use_container_width=True)
            if not HAS_ARIMA:
                st.info("ARIMA not available; used linear trend extrapolation as fallback. Install `statsmodels` for ARIMA forecasting.")

# ----------------------------------------------------------
# NAV: CONTROL CENTER
# ----------------------------------------------------------
elif nav == "Control Center":
    st.header("🕹️ Control Center — Simulate Actions & See Effects")
    st.write("Simulate capacitor banks, load shedding, and view immediate impact on feeder metrics. This only affects the simulation layer (non-destructive).")

    with st.form("control_form"):
        cap_on = st.checkbox("Capacitor Bank ON", value=st.session_state.control_state.get("cap_on", False))
        cap_eff = st.slider("Capacitor effectiveness (% reactive reduction)", 0, 100, int(st.session_state.control_state.get("cap_effect_pct", 10)))
        shed_target = st.selectbox("Load shed target", ["All", "This Feeder"])
        shed_pct = st.slider("Load shed percentage (%)", 0, 100, int(st.session_state.control_state.get("shed_pct", 0)))
        submitted = st.form_submit_button("Apply Simulation")
    if submitted:
        st.session_state.control_state.update({"cap_on": cap_on, "cap_effect_pct": cap_eff, "shed_pct": shed_pct, "shed_target": shed_target})
        st.success("Simulation updated. Switch to a Feeder tab or Home to see effects.")

    st.markdown("---")
    st.subheader("Quick actions")
    col1, col2 = st.columns(2)
    if col1.button("Clear Alerts"):
        clear_alerts()
        st.success("Alerts cleared.")
    if col2.button("Reset Simulation State"):
        st.session_state.control_state = {"cap_on": False, "cap_effect_pct": 10.0, "shed_pct": 0.0, "shed_target": "All"}
        st.success("Simulation reset.")

    st.markdown("### Current Simulation State")
    st.json(st.session_state.control_state)

# ----------------------------------------------------------
# NAV: ALERTS
# ----------------------------------------------------------
elif nav == "Alerts":
    st.header("🔔 Alerts Log & Management")
    alerts = get_alerts()
    if not alerts:
        st.info("No alerts logged.")
    else:
        for a in alerts[::-1]:
            if a["severity"] == "error":
                st.error(f"{a['time']}  —  {a['msg']}")
            elif a["severity"] == "warning":
                st.warning(f"{a['time']}  —  {a['msg']}")
            else:
                st.info(f"{a['time']}  —  {a['msg']}")
        st.download_button("Download Alerts (JSON)", json.dumps(alerts, indent=2), "alerts.json")

    if st.button("Clear Alerts Log"):
        clear_alerts()
        st.experimental_rerun()

# ----------------------------------------------------------
# NAV: SETTINGS
# ----------------------------------------------------------
elif nav == "Settings":
    st.header("⚙️ Settings & Data Export")
    st.write("Download current datasets or upload new CSVs. Uploads replace in-memory dataset (will not persist between runs unless you save files).")
    cols = st.columns(2)
    if cols[0].button("Download Total Dataset (CSV)"):
        csv = data_store['total'].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "total_dataset.csv")
    if cols[1].button("Download Feeders (zipped CSVs)"):
        # create zip in-memory
        import zipfile, tempfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for key, fname in [("total", CSV_PATH), ("A_p", FEEDER_A_P), ("A_q", FEEDER_A_Q), ("B_p", FEEDER_B_P), ("B_q", FEEDER_B_Q), ("C_p", FEEDER_C_P), ("C_q", FEEDER_C_Q)]:
                if os.path.exists(fname):
                    z.write(fname)
        buf.seek(0)
        st.download_button("Download ZIP", buf, "datasets.zip")

    st.markdown("---")
    st.subheader("Upload new dataset (optional)")
    uploaded = st.file_uploader("Upload total CSV to replace", type=['csv'])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            data_store['total'] = new_df
            st.success("Total dataset uploaded to session (non-persistent).")
        except Exception as e:
            st.error(f"Failed to load uploaded CSV: {e}")

# ----------------------------------------------------------
# FALLBACK (safety)
# ----------------------------------------------------------
else:
    st.info("Select a page from the sidebar to begin.")

# End of dashboard_pro_v2.py
