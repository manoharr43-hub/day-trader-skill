"""India NSE Trading Desk — Yahoo Finance edition. No FYERS integration."""
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="India NSE 18-Agent Trading Desk", page_icon="🇮🇳", layout="wide", initial_sidebar_state="expanded")

from nse_smart_money import render_smart_money_page

with st.sidebar:
    st.title("🇮🇳 INDIA NSE DESK")
    st.caption("Yahoo Finance · No FYERS")
    page = st.radio("Navigation", ["🧠 18-Agent Smart Money Scanner", "📘 Original project modules"], key="main_page")
    st.divider()
    st.caption("Market: NSE India · Intraday + Swing · Equity + F&O")

if page == "🧠 18-Agent Smart Money Scanner":
    render_smart_money_page()
else:
    st.title("Original Day-Trader Skill Modules")
    st.info("The uploaded project’s original trader-* folders are preserved in this ZIP. They were designed around US-market workflows and are not silently presented as NSE-ready. The new NSE 18-agent dashboard is the India-focused module.")
    modules = [
        ("trader-scanner", "Chart scanner source"), ("trader-news", "News monitor source"),
        ("trader-entry", "Entry timing source"), ("trader-exit", "Exit planner source"),
        ("trader-timemachine", "Historical simulation source"), ("trader-risk", "Risk manager source"),
        ("trader-budget", "Position sizing source"), ("trader-broker", "Broker ticket source"),
        ("trader-alerts", "Price alerts source"), ("trader-analyst", "Earnings analyst source"),
        ("trader-pelosi", "US Congress trade tracker — not applicable to NSE"),
    ]
    for name, desc in modules:
        st.write(f"**{name}** — {desc}")
    st.warning("US-specific scripts require separate India-specific adaptation before execution. No broker orders are sent by this dashboard.")
