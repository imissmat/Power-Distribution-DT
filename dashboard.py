# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

# -------------------------
# Config / Load data
# -------------------------
st.set_page_config(layout="wide", page_title="Digital Twin - Live Plots")
st.title("Digital Twin — Live Plots")

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

# -------------------------
# Placeholders / Layout
# -------------------------
col_left, col_right = st.columns([1,1])
pwr_pl = col_left.empty()    # Active & Reactive power
volt_pl = col_right.empty()  # Voltage + TapA
kwh_pl = st.container().empty()  # Bus2002_kWh (full width below)

# -------------------------
# Live streaming loop
# -------------------------
times, P_vals, Q_vals, V_vals, Tap_vals, kWh_vals = [], [], [], [], [], []

try:
    for idx, row in df.iterrows():
        # append latest row
        times.append(row['Time'])
        P_vals.append(row['Active_Power'])
        Q_vals.append(row['Reac_Power'])
        V_vals.append(row['Bus2002_Voltage'])
        Tap_vals.append(row['TapA'])
        kWh_vals.append(row['Bus2002_kWh'])

        # --- Plot 1: Active & Reactive Power (compact) ---
        fig1, ax1 = plt.subplots(figsize=(6, 3))  # smaller size
        ax1.plot(times, P_vals, label='Active Power (P)')
        ax1.plot(times, Q_vals, label='Reactive Power (Q)')
        ax1.set_title("Active & Reactive Power")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Power")
        ax1.legend(loc='upper right')
        ax1.grid(True)
        fig1.tight_layout()
        pwr_pl.pyplot(fig1)
        plt.close(fig1)

        # --- Plot 2: Bus2002 Voltage + TapA (compact) ---
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(times, V_vals, label='Bus2002 Voltage (p.u.)')
        ax2.axhline(0.95, linestyle='--', linewidth=1, label='0.95 pu')
        ax2.axhline(1.05, linestyle='--', linewidth=1, label='1.05 pu')
        ax2.set_title("Bus2002 Voltage (p.u.)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Voltage (p.u.)")
        ax2.grid(True)
        ax2.legend(loc='upper left')

        # twin axis for TapA (step)
        ax2_t = ax2.twinx()
        ax2_t.step(times, Tap_vals, where='post', label='TapA', linestyle='--', marker='o')
        ax2_t.set_ylabel("Tap Position")
        # set reasonable tap limits based on dataset
        tap_min, tap_max = int(df['TapA'].min()), int(df['TapA'].max())
        ax2_t.set_ylim(tap_min - 1, tap_max + 1)
        ax2_t.legend(loc='upper right')

        fig2.tight_layout()
        volt_pl.pyplot(fig2)
        plt.close(fig2)

        # --- Plot 3: Bus2002_kWh (full-width, compact height) ---
        fig3, ax3 = plt.subplots(figsize=(12, 3))
        ax3.plot(times, kWh_vals, label='Bus2002_kWh')
        ax3.set_title("Bus2002 Energy Consumption (kWh)")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("kWh")
        ax3.grid(True)
        ax3.legend(loc='upper right')
        fig3.tight_layout()
        kwh_pl.pyplot(fig3)
        plt.close(fig3)

        # send next timestamp after 2 seconds (live effect)
        time.sleep(2)

except KeyboardInterrupt:
    # allows graceful stop if running locally and user interrupts
    st.write("Live stream stopped by user.")
