"""India NSE Trading Desk - NSE AI PRO V18 (Yahoo Finance only, no broker login).

Entry point for Streamlit (local or Streamlit Cloud):  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="NSE AI PRO V18", page_icon="🇮🇳", layout="wide", initial_sidebar_state="expanded")

from nse_smart_money import render_smart_money_page
from nsepro import ui_ml, ui_reports, ui_scanner, ui_search

PAGES = [
    "🧠 18-Agent Smart Money Scanner",
    "🚀 Big Movement Scanner",
    "🔍 AI Stock Search",
    "🤖 AI Model (ML)",
    "📋 Reports",
    "📘 Original project modules",
]

with st.sidebar:
    st.title("🇮🇳 NSE AI PRO V18")
    st.caption("Yahoo Finance data · no broker login")
    page = st.radio("Navigation", PAGES, key="main_page")
    st.divider()
    st.caption("Market: NSE India · Intraday + Swing · Equity + F&O")
    st.caption("Data is unofficial and may be delayed. Not financial advice. This app never places orders.")


def original_modules():
    st.title("Original Day-Trader Skill Modules")
    st.info("The uploaded project's original trader-* folders are preserved in this ZIP. They were designed around US-market workflows and are not silently presented as NSE-ready. The NSE dashboards in this app are the India-focused modules.")
    modules = [
        ("trader-scanner", "Chart scanner source"), ("trader-news", "News monitor source"),
        ("trader-entry", "Entry timing source"), ("trader-exit", "Exit planner source"),
        ("trader-timemachine", "Historical simulation source"), ("trader-risk", "Risk manager source"),
        ("trader-budget", "Position sizing source"), ("trader-broker", "Broker ticket source"),
        ("trader-alerts", "Price alerts source"), ("trader-analyst", "Earnings analyst source"),
        ("trader-pelosi", "US Congress trade tracker, not applicable to NSE"),
    ]
    for name, desc in modules:
        st.write(f"**{name}**: {desc}")
    st.warning("US-specific scripts require separate India-specific adaptation before execution. No broker orders are sent by this dashboard.")


if page == PAGES[0]:
    render_smart_money_page()
elif page == PAGES[1]:
    ui_scanner.render_page()
elif page == PAGES[2]:
    ui_search.render_page()
elif page == PAGES[3]:
    ui_ml.render_page()
elif page == PAGES[4]:
    ui_reports.render_page()
else:
    original_modules()
