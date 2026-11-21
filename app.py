"""Streamlit front end for the 11 tools in day-trader-skill.
Run from this folder: streamlit run app.py
The underlying tools remain the source of trading logic; this file only provides a UI.
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

st.set_page_config(page_title="Day Trader Desk", page_icon="📈", layout="wide")

TOOLS = {
    "Scanner": ROOT / "trader-scanner" / "scan.py",
    "News Monitor": ROOT / "trader-news" / "news.py",
    "Entry Timer": ROOT / "trader-entry" / "entry.py",
    "Exit Planner": ROOT / "trader-exit" / "exit.py",
    "Time Machine": ROOT / "trader-timemachine" / "timemachine.py",
    "Budget Manager": ROOT / "trader-budget" / "size.py",
    "Risk Manager": ROOT / "trader-risk" / "check.py",
    "Order Tickets": ROOT / "trader-broker" / "ticket.py",
    "Price Alerts": ROOT / "trader-alerts" / "alerts.py",
    "Earnings Analyst": ROOT / "trader-analyst" / "facts.py",
    "Congress Trades": ROOT / "trader-pelosi" / "ptr.py",
}

st.title("📈 Day Trader Desk")
st.caption("Streamlit interface for the uploaded Day Trader skill pack · Yahoo Finance/public data · Paper-first workflow")

missing = [name for name, path in TOOLS.items() if not path.is_file()]
if missing:
    st.error("Some tool files are missing. Keep app.py beside all trader-* folders. Missing: " + ", ".join(missing))
    st.stop()


def run_tool(name: str, args: list[str], timeout: int = 240):
    """Run one repository CLI tool without invoking a shell."""
    script = TOOLS[name]
    cmd = [sys.executable, str(script), *[str(x) for x in args if x is not None and str(x) != ""]]
    with st.spinner(f"Running {name}..."):
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            st.error(f"{name} timed out after {timeout} seconds. Try a smaller scan or try again later.")
            return
        except Exception as exc:
            st.exception(exc)
            return
    output = (proc.stdout or "").strip()
    errors = (proc.stderr or "").strip()
    if proc.returncode == 0:
        st.success(f"{name} completed")
    else:
        st.warning(f"{name} finished with exit code {proc.returncode}")
    if output:
        st.code(output, language="text")
    if errors:
        with st.expander("Diagnostic / error output", expanded=proc.returncode != 0):
            st.code(errors, language="text")
    if not output and not errors:
        st.info("The tool returned no console output. Check its trader/ output files.")


def symbol_input(label="Ticker symbols", default="AAPL", key="symbols"):
    return st.text_input(label, default, key=key).strip()

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose module", list(TOOLS.keys()), index=0)
    st.divider()
    st.caption("Project folder")
    st.code(str(ROOT), language="text")
    st.caption("The scripts use public market-data sources. Data can be delayed, incomplete, or rate-limited.")
    if st.button("Create trader data folders", use_container_width=True):
        for folder in ["trader", "trader/state", "trader/state/ptr-pdfs"]:
            (ROOT / folder).mkdir(parents=True, exist_ok=True)
        st.success("Folders created")

# Shared module panels
if page == "Scanner":
    st.subheader("🔎 Chart Scanner")
    st.write("Daily setups and intraday opening-range breakouts. The scanner defaults to the repository's US-stock universe.")
    tab1, tab2, tab3 = st.tabs(["Daily scan", "Opening-range breakout", "Build universe"])
    with tab1:
        symbols = symbol_input("Optional tickers (comma-separated; leave blank to use universe)", "", "daily_symbols")
        limit = st.number_input("Maximum symbols to scan (0 = tool default)", min_value=0, max_value=10000, value=250, step=50)
        show = st.number_input("Maximum hits to print", min_value=1, max_value=500, value=50)
        if st.button("Run daily scan", type="primary"):
            args = ["daily", "--show", str(show)]
            if symbols: args += ["--symbols", symbols]
            if limit: args += ["--limit", str(int(limit))]
            run_tool("Scanner", args, timeout=900)
    with tab2:
        symbols = symbol_input("Optional tickers (comma-separated; leave blank to use universe)", "", "orb_symbols")
        at = st.text_input("New York time cut-off (HH:MM, optional)", "", key="orb_at")
        max_confirm = st.number_input("Maximum charts for confirmation", min_value=0, max_value=5000, value=250, step=50)
        if st.button("Run ORB scan", type="primary"):
            args = ["orb", "--show", "100", "--max-confirm", str(int(max_confirm))]
            if symbols: args += ["--symbols", symbols]
            if at: args += ["--at", at]
            run_tool("Scanner", args, timeout=900)
    with tab3:
        st.info("Downloads the Nasdaq Trader symbol lists used by this project's scanner (US listings, not NSE symbols).")
        if st.button("Download / update stock universe"):
            run_tool("Scanner", ["universe"], timeout=180)

elif page == "News Monitor":
    st.subheader("📰 Breaking News Monitor")
    tickers = symbol_input("Tickers (space-separated)", "AAPL TSLA", "news_tickers")
    hours = st.slider("Look-back hours", 1, 168, 48)
    no_sec = st.checkbox("Headlines only (skip SEC filings)", value=False)
    dry = st.checkbox("Dry run (do not update seen-news state)", value=True)
    if st.button("Fetch news", type="primary"):
        args = ["--tickers", *tickers.split(), "--hours", str(hours)]
        if no_sec: args.append("--no-sec")
        if dry: args.append("--dry-run")
        run_tool(page, args, timeout=240)

elif page == "Entry Timer":
    st.subheader("⏱️ Entry Timer")
    ticker = st.text_input("Ticker", "TSLA", key="entry_symbol").strip().upper()
    setup = st.selectbox("Setup", ["orb", "level"])
    interval = st.selectbox("Candle interval", ["5m", "1m"])
    level = st.number_input("Trigger level (only for level setup)", min_value=0.0, value=0.0, step=0.1)
    if st.button("Check entry status", type="primary"):
        args = ["check", ticker, "--setup", setup, "--interval", interval, "--json"]
        if setup == "level" and level > 0: args += ["--level", str(level)]
        run_tool(page, args, timeout=180)
    st.caption("Watchlist check reads trader/watchlist.json. The continuous watch command is intentionally not exposed in the browser UI.")
    if st.button("Check scanner watchlist"):
        run_tool(page, ["watchlist", "--json"], timeout=240)

elif page == "Exit Planner":
    st.subheader("🛑 Exit Planner")
    st.caption("Create a stop/target plan. This writes a plan file; it does not place an order.")
    ticker = st.text_input("Ticker", "TSLA", key="exit_symbol").strip().upper()
    entry = st.number_input("Entry price", min_value=0.0, value=0.0, step=0.01)
    stop = st.text_input("Stop rule", "or-mid", help="Examples: or-mid, or-low, pct:1, price:100")
    target = st.selectbox("Target rule", ["1R", "1.5R", "2R"] , index=2)
    side = st.selectbox("Side", ["buy", "sell"])
    if st.button("Build exit plan", type="primary"):
        if entry <= 0: st.error("Enter a valid entry price.")
        else: run_tool(page, ["plan", "--symbol", ticker, "--entry", str(entry), "--stop", stop,
                              "--target", target, "--side", side, "--write", "--json"], timeout=180)
    if st.button("Show current plan"):
        run_tool(page, ["show", "--json"], timeout=60)

elif page == "Time Machine":
    st.subheader("🕰️ Historical Replay / Time Machine")
    st.warning("Simulation and historical results are not guarantees of future performance. A saved trade plan is required unless you supply trade parameters in the CLI.")
    if st.button("Run replay from trader/plan.json", type="primary"):
        run_tool(page, ["run", "--json", "--write"], timeout=600)
    if st.button("Show saved replay chart"):
        run_tool(page, ["chart", "--from"], timeout=120)

elif page == "Budget Manager":
    st.subheader("💰 Position Sizing")
    budget = st.number_input("Optional account budget override", min_value=0.0, value=0.0, step=100.0)
    if st.button("Calculate position size", type="primary"):
        args = ["--json", "--write"]
        if budget > 0: args += ["--budget", str(budget)]
        run_tool(page, args, timeout=120)

elif page == "Risk Manager":
    st.subheader("🛡️ Risk Checks")
    st.caption("Checks proposed orders against config, portfolio, prices and daily limits.")
    no_write = st.checkbox("Do not save risk verdict", value=True)
    if st.button("Run risk check", type="primary"):
        args = ["--json"] + (["--no-write"] if no_write else [])
        run_tool(page, args, timeout=120)

elif page == "Order Tickets":
    st.subheader("🎫 Broker Order Tickets")
    st.warning("This page only displays and records ticket state. No live broker connection or automatic order placement is performed by this Streamlit app.")
    if st.button("Show prepared tickets", type="primary"):
        run_tool(page, ["show", "--json"], timeout=60)
    st.caption("The original tool requires a separate explicit approval flow and a separately configured broker connector before any live order. Do not treat a displayed ticket as an executed trade.")

elif page == "Price Alerts":
    st.subheader("🔔 Price Alerts")
    tab1, tab2 = st.tabs(["Market clock", "Check existing alerts"])
    with tab1:
        if st.button("Check market clock", type="primary"):
            run_tool(page, ["clock", "--json"], timeout=60)
    with tab2:
        if st.button("Check alert rules", type="primary"):
            run_tool(page, ["check", "--json"], timeout=180)
        if st.button("List alert rules"):
            run_tool(page, ["list"], timeout=60)

elif page == "Earnings Analyst":
    st.subheader("📑 Company / Earnings Analyst")
    tickers = st.text_input("Ticker symbols (space-separated)", "AAPL", key="analyst_tickers").strip()
    quarters = st.slider("Quarters", 1, 12, 6)
    if st.button("Analyze company filings and facts", type="primary"):
        run_tool(page, [*tickers.split(), "--quarters", str(quarters), "--json"], timeout=240)

elif page == "Congress Trades":
    st.subheader("🏛️ Congressional Trading Disclosures")
    year = st.number_input("Filing year", min_value=2012, max_value=2035, value=2026, step=1)
    member = st.text_input("Member name filter (optional)", "pelosi")
    limit = st.number_input("Maximum filings", min_value=1, max_value=500, value=50)
    dry = st.checkbox("Do not update seen-filings state", value=True)
    if st.button("Fetch disclosure index", type="primary"):
        args = ["index", "--year", str(int(year)), "--limit", str(int(limit)), "--json"]
        if member.strip(): args += ["--member", member.strip()]
        if dry: args.append("--dry-run")
        run_tool(page, args, timeout=240)

st.divider()
st.caption("Important: This repository describes a US-market workflow (Nasdaq Trader universe, US market session, SEC EDGAR and House Clerk). It is not an NSE/FYERS app. Review all data and risk settings before making decisions.")
