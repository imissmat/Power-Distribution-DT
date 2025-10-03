# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# -------------------------
# Config / Load data
# -------------------------
st.set_page_config(layout="wide", page_title="Insurance Descriptive Analytics")
# Placeholder for company logo (replace with actual image path/URL later)
st.image("Company_Logo.jpg", width=150)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'])
    # average the three phase voltages for bus2002 if present
    if all(c in df.columns for c in ['bus2002.1','bus2002.2','bus2002.3']):
        df['Bus2002_Voltage'] = df[['bus2002.1','bus2002.2','bus2002.3']].mean(axis=1)
    else:
        # fallback if a single column exists
        df['Bus2002_Voltage'] = df.get('Bus2002_Voltage', df.get('bus2002', pd.NA))
    return df

DATA_PATH = "New_Collected_Data.csv"
df = load_data(DATA_PATH)

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Feeder A"])

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("🏠 Home")
    
    # KPIs for Active and Reactive Power
    col_kpi1, col_kpi2 = st.columns(2)
    active_kpi = col_kpi1.empty()
    reactive_kpi = col_kpi2.empty()
    
    # Plots placeholders
    col_plot1, col_plot2 = st.columns(2)
    pwr_active_pl = col_plot1.empty()
    pwr_reactive_pl = col_plot2.empty()
    
    # Live streaming loop for Home
    times, P_vals, Q_vals = [], [], []
    
    try:
        for idx, row in df.iterrows():
            # Append latest row data
            times.append(row['Time'])
            P_vals.append(row['Active_Power'])
            Q_vals.append(row['Reac_Power'])
            
            # Update KPIs with latest values
            active_kpi.metric("Total Active Power", f"{P_vals[-1]:.2f} kW")
            reactive_kpi.metric("Total Reactive Power", f"{Q_vals[-1]:.2f} kVAR")
            
            # Plot 1: Active Power over time
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(times, P_vals, label='Active Power (P)', color='blue')
            ax1.set_title("Total Active Power Over Time")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Power (kW)")
            ax1.legend(loc='upper right')
            ax1.grid(True)
            fig1.tight_layout()
            pwr_active_pl.pyplot(fig1)
            plt.close(fig1)
            
            # Plot 2: Reactive Power over time
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(times, Q_vals, label='Reactive Power (Q)', color='red')
            ax2.set_title("Total Reactive Power Over Time")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Power (kVAR)")
            ax2.legend(loc='upper right')
            ax2.grid(True)
            fig2.tight_layout()
            pwr_reactive_pl.pyplot(fig2)
            plt.close(fig2)
            
            # Simulate live update delay
            time.sleep(2)
            
    except KeyboardInterrupt:
        st.write("Live stream stopped by user.")

# -------------------------
# Feeder A Page (Bus 2002)
# -------------------------
elif page == "Feeder A":
    st.title("🔌 Feeder A - Bus 2002")
    
    # Layout for plots
    col_left, col_right = st.columns([1, 1])
    kwh_pl = st.container().empty()  # Full width for kWh
    volt_pl = col_left.empty()       # Voltage in left
    tap_pl = col_right.empty()       # Tap in right
    
    # Live streaming loop for Feeder A
    times, V_vals, Tap_vals, kWh_vals = [], [], [], []
    
    try:
        for idx, row in df.iterrows():
            # Append latest row data
            times.append(row['Time'])
            V_vals.append(row['Bus2002_Voltage'])
            Tap_vals.append(row['TapA'])
            kWh_vals.append(row['Bus2002_kWh'])
            
            # Plot 1: Bus2002_kWh over time (full width)
            fig_kwh, ax_kwh = plt.subplots(figsize=(12, 4))
            ax_kwh.plot(times, kWh_vals, label='Bus2002_kWh', color='green')
            ax_kwh.set_title("Bus2002 Energy Consumption (kWh) Over Time")
            ax_kwh.set_xlabel("Time")
            ax_kwh.set_ylabel("kWh")
            ax_kwh.legend(loc='upper right')
            ax_kwh.grid(True)
            fig_kwh.tight_layout()
            kwh_pl.pyplot(fig_kwh)
            plt.close(fig_kwh)
            
            # Plot 2: Bus2002 Voltage over time with reference lines
            fig_volt, ax_volt = plt.subplots(figsize=(6, 4))
            ax_volt.plot(times, V_vals, label='Bus2002 Voltage (p.u.)', color='purple')
            ax_volt.axhline(0.95, linestyle='--', linewidth=2, color='orange', label='0.95 pu')
            ax_volt.axhline(1.05, linestyle='--', linewidth=2, color='orange', label='1.05 pu')
            ax_volt.set_title("Bus2002 Voltage (p.u.) Over Time")
            ax_volt.set_xlabel("Time")
            ax_volt.set_ylabel("Voltage (p.u.)")
            ax_volt.legend(loc='upper left')
            ax_volt.grid(True)
            fig_volt.tight_layout()
            volt_pl.pyplot(fig_volt)
            plt.close(fig_volt)
            
            # Plot 3: TapA changing over time
            fig_tap, ax_tap = plt.subplots(figsize=(6, 4))
            ax_tap.step(times, Tap_vals, where='post', label='TapA', linestyle='--', marker='o', color='brown')
            ax_tap.set_title("TapA Position Over Time")
            ax_tap.set_xlabel("Time")
            ax_tap.set_ylabel("Tap Position")
            ax_tap.legend(loc='upper right')
            ax_tap.grid(True)
            # Set y-limits based on data range
            tap_min, tap_max = min(Tap_vals), max(Tap_vals)
            ax_tap.set_ylim(tap_min - 1, tap_max + 1)
            fig_tap.tight_layout()
            tap_pl.pyplot(fig_tap)
            plt.close(fig_tap)
            
            # Simulate live update delay
            time.sleep(2)
            
    except KeyboardInterrupt:
        st.write("Live stream stopped by user.")