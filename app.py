"""Indian NSE market dashboard built on Yahoo Finance data.
Run: streamlit run app.py
This is a research/paper-trading dashboard; it does not connect to a broker or place orders.
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent
IST = "Asia/Kolkata"

st.set_page_config(page_title="India NSE AI Trading Desk", page_icon="🇮🇳", layout="wide")

NSE_UNIVERSE = {
    "RELIANCE": "Reliance Industries", "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India", "INFY": "Infosys", "TCS": "Tata Consultancy Services",
    "ITC": "ITC", "BHARTIARTL": "Bharti Airtel", "LT": "Larsen & Toubro", "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank", "HINDUNILVR": "Hindustan Unilever", "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki", "SUNPHARMA": "Sun Pharma", "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel", "NTPC": "NTPC", "POWERGRID": "Power Grid", "M&M": "Mahindra & Mahindra",
    "ULTRACEMCO": "UltraTech Cement", "ONGC": "ONGC", "WIPRO": "Wipro", "HCLTECH": "HCL Technologies",
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports", "COALINDIA": "Coal India",
    "JSWSTEEL": "JSW Steel", "TITAN": "Titan", "ASIANPAINT": "Asian Paints", "GRASIM": "Grasim",
    "TECHM": "Tech Mahindra", "DRREDDY": "Dr Reddy's Labs", "CIPLA": "Cipla", "DIVISLAB": "Divi's Labs",
    "EICHERMOT": "Eicher Motors", "HEROMOTOCO": "Hero MotoCorp", "BAJAJ-AUTO": "Bajaj Auto",
    "APOLLOHOSP": "Apollo Hospitals", "BPCL": "BPCL", "IOC": "Indian Oil", "HINDALCO": "Hindalco",
    "INDUSINDBK": "IndusInd Bank", "BEL": "Bharat Electronics", "TRENT": "Trent", "ZOMATO": "Eternal",
    "PIDILITIND": "Pidilite", "SBILIFE": "SBI Life", "HDFCLIFE": "HDFC Life", "SHRIRAMFIN": "Shriram Finance",
}
INDEXES = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN", "INDIA VIX": "^INDIAVIX"}


def yf_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.startswith("NSE:"):
        s = s.split(":", 1)[1]
    if s.endswith("-EQ"):
        s = s[:-3]
    if s.endswith(".NS") or s.startswith("^") or s.endswith(".BO"):
        return s
    return s + ".NS"


@st.cache_data(ttl=120, show_spinner=False)
def load_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    ticker = yf_symbol(symbol)
    frame = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                        progress=False, threads=False, group_by="column")
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).title() for c in frame.columns]
    required = ["Open", "High", "Low", "Close"]
    if not all(c in frame.columns for c in required):
        return pd.DataFrame()
    if "Volume" not in frame.columns:
        frame["Volume"] = 0
    frame = frame[~frame.index.duplicated(keep="last")].copy()
    for c in required + ["Volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.dropna(subset=required)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low, volume = x.Close, x.High, x.Low, x.Volume
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()
    x["EMA200"] = close.ewm(span=200, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = 100 - 100 / (1 + rs)
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["RVOL"] = volume / volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    x["High20"] = high.shift(1).rolling(20).max()
    x["Low20"] = low.shift(1).rolling(20).min()
    x["Return1D%"] = close.pct_change() * 100
    return x


def signal_for(row: pd.Series) -> tuple[str, str]:
    close = row.get("Close", np.nan)
    e20, e50, rsi = row.get("EMA20", np.nan), row.get("EMA50", np.nan), row.get("RSI14", np.nan)
    high20 = row.get("High20", np.nan)
    if pd.isna(close) or pd.isna(e20) or pd.isna(e50) or pd.isna(rsi):
        return "INSUFFICIENT DATA", "Need more candles"
    if pd.notna(high20) and close > high20 and e20 > e50 and rsi >= 55:
        return "BREAKOUT WATCH", "Close above prior 20-candle high; confirm volume and next candle"
    if close > e20 > e50 and 50 <= rsi < 70:
        return "BULLISH TREND", "Price above EMA20/EMA50 with positive RSI"
    if close < e20 < e50 and rsi <= 45:
        return "BEARISH TREND", "Price below EMA20/EMA50 with weak RSI"
    if pd.notna(row.get("Low20")) and close < row.get("Low20") and e20 < e50:
        return "BREAKDOWN WATCH", "Close below prior 20-candle low"
    return "WAIT", "No clear trend alignment"


def fmt_price(v):
    return "—" if pd.isna(v) else f"₹{float(v):,.2f}"


def render_market_dashboard():
    st.subheader("🇮🇳 Indian Market Dashboard")
    cols = st.columns(4)
    for col, (name, ticker) in zip(cols, INDEXES.items()):
        try:
            d = load_history(ticker, "5d", "1d")
            if d.empty:
                col.metric(name, "No data")
            else:
                last = float(d.Close.iloc[-1]); prev = float(d.Close.iloc[-2]) if len(d) > 1 else last
                col.metric(name, f"{last:,.2f}", f"{last-prev:+,.2f} ({(last/prev-1)*100:+.2f}%)" if prev else None)
        except Exception as e:
            col.metric(name, "Unavailable")
    st.caption("Yahoo Finance public data. Index quotes and stock prices may be delayed, incomplete or unavailable.")
    st.markdown("**Quick watchlist**")
    symbols = st.multiselect("Choose NSE stocks", list(NSE_UNIVERSE), default=["RELIANCE", "HDFCBANK", "SBIN", "INFY", "TCS"])
    if st.button("Load watchlist quotes", type="primary"):
        rows=[]
        bar=st.progress(0)
        for i,s in enumerate(symbols):
            d=load_history(s,"5d","1d")
            if not d.empty:
                last=float(d.Close.iloc[-1]); prev=float(d.Close.iloc[-2]) if len(d)>1 else last
                rows.append({"Symbol":s,"Company":NSE_UNIVERSE.get(s,""),"LTP":last,"Change %":(last/prev-1)*100 if prev else 0})
            bar.progress((i+1)/max(len(symbols),1))
        if rows: st.dataframe(pd.DataFrame(rows).style.format({"LTP":"₹{:,.2f}","Change %":"{:+.2f}%"}), use_container_width=True, hide_index=True)
        else: st.warning("No quote data returned. Try again later or verify Yahoo ticker availability.")


def render_scanner():
    st.subheader("🔎 NSE Stock Scanner")
    st.write("Technical screening using EMA20/50, RSI, 20-candle breakout and relative volume. This is a screening signal, not a prediction.")
    mode=st.radio("Universe",["NIFTY 50-style watchlist (curated sample)","Enter my own symbols"],horizontal=True)
    raw=st.text_input("NSE symbols separated by comma", "RELIANCE, HDFCBANK, SBIN, INFY, TCS, ICICIBANK" if mode.startswith("Enter") else "")
    symbols=[s.strip().upper() for s in raw.split(",") if s.strip()] if mode.startswith("Enter") else list(NSE_UNIVERSE)[:30]
    period=st.selectbox("History",["3mo","6mo","1y","2y"],index=1)
    if st.button("Run NSE scan",type="primary"):
        rows=[]; progress=st.progress(0); status=st.empty()
        for i,s in enumerate(symbols):
            status.write(f"Checking {s} ({i+1}/{len(symbols)})")
            d=load_history(s,period,"1d")
            if not d.empty:
                x=indicators(d); r=x.iloc[-1]; sig,reason=signal_for(r)
                rows.append({"Symbol":s,"LTP":float(r.Close),"Change %":float(r.get("Return1D%",np.nan)),"RSI":float(r.get("RSI14",np.nan)),"RVOL":float(r.get("RVOL",np.nan)),"EMA20":float(r.get("EMA20",np.nan)),"EMA50":float(r.get("EMA50",np.nan)),"Signal":sig,"Reason":reason})
            progress.progress((i+1)/max(len(symbols),1))
        status.empty()
        if rows:
            out=pd.DataFrame(rows).sort_values(["Signal","RVOL"],ascending=[True,False])
            st.dataframe(out.style.format({"LTP":"₹{:,.2f}","Change %":"{:+.2f}%","RSI":"{:.1f}","RVOL":"{:.2f}x","EMA20":"₹{:,.2f}","EMA50":"₹{:,.2f}"}),use_container_width=True,hide_index=True)
            st.download_button("Download scan CSV",out.to_csv(index=False).encode(),"nse_scan.csv","text/csv")
        else: st.warning("No data returned for the selected symbols.")


def render_analysis():
    st.subheader("📊 Individual Stock Analysis")
    symbol=st.text_input("NSE symbol (without .NS)","RELIANCE",key="analysis_symbol").strip().upper()
    tf=st.selectbox("Timeframe",["1d","5d","1mo","3mo","6mo","1y","2y"],index=4)
    interval=st.selectbox("Candle interval",["1d","1h","30m","15m","5m"],index=0)
    if st.button("Analyze stock",type="primary"):
        d=load_history(symbol,tf,interval)
        if d.empty: st.error("Data unavailable. Check symbol (example RELIANCE) and try a supported interval.")
        else:
            x=indicators(d); r=x.iloc[-1]; sig,reason=signal_for(r)
            a,b,c,dcol=st.columns(4)
            a.metric("Latest close",fmt_price(r.Close)); b.metric("RSI 14",f"{r.RSI14:.1f}" if pd.notna(r.RSI14) else "—")
            c.metric("Relative volume",f"{r.RVOL:.2f}x" if pd.notna(r.RVOL) else "—"); dcol.metric("Technical status",sig)
            st.info(reason)
            st.line_chart(x[["Close","EMA20","EMA50"]].dropna())
            st.dataframe(x.tail(20).sort_index(ascending=False),use_container_width=True)


def render_entry_exit():
    st.subheader("🎯 Entry / Exit Planner (Paper plan)")
    c1,c2,c3=st.columns(3)
    symbol=c1.text_input("Symbol","RELIANCE").upper().strip()
    side=c2.selectbox("Side",["BUY","SELL"])
    entry=c3.number_input("Entry price (₹)",min_value=0.0,value=1000.0,step=0.5)
    stop_pct=st.slider("Stop-loss distance (%)",0.25,10.0,2.0,0.25)
    rr=st.selectbox("Risk : Reward",[1.0,1.5,2.0,3.0],index=2)
    if entry>0:
        stop=entry*(1-stop_pct/100) if side=="BUY" else entry*(1+stop_pct/100)
        target=entry+(entry-stop)*rr if side=="BUY" else entry-(stop-entry)*rr
        a,b,c=st.columns(3); a.metric("Entry",fmt_price(entry)); b.metric("Stop",fmt_price(stop)); c.metric("Target",fmt_price(target))
        st.caption(f"{symbol} · {side} · Risk/reward 1:{rr:g}. Plan only; no order is sent.")


def render_position_size():
    st.subheader("💰 Position Size & Risk Calculator")
    c1,c2,c3=st.columns(3)
    capital=c1.number_input("Trading capital (₹)",min_value=0.0,value=100000.0,step=5000.0)
    risk_pct=c2.number_input("Risk per trade (%)",min_value=0.1,max_value=5.0,value=1.0,step=0.1)
    entry=c3.number_input("Entry price (₹)",min_value=0.01,value=1000.0,step=1.0)
    stop=st.number_input("Stop-loss price (₹)",min_value=0.0,value=980.0,step=1.0)
    if entry>0 and stop>0 and entry!=stop:
        risk_amount=capital*risk_pct/100; per_share=abs(entry-stop); qty=math.floor(risk_amount/per_share)
        st.metric("Maximum quantity (shares)",f"{qty:,}")
        st.write(f"Maximum planned loss: **₹{qty*per_share:,.2f}** · Position value: **₹{qty*entry:,.2f}**")
    else: st.warning("Enter valid entry and stop-loss prices; they must be different.")


def render_backtest():
    st.subheader("🧪 Historical Backtest — EMA crossover")
    st.caption("Simple educational backtest with no brokerage, taxes, slippage or survivorship-bias adjustment. Past results do not predict future returns.")
    symbol=st.text_input("NSE symbol","SBIN",key="bt_symbol").upper().strip()
    period=st.selectbox("Backtest history",["1y","2y","5y","10y"],index=1)
    fast=st.number_input("Fast EMA",5,100,20); slow=st.number_input("Slow EMA",10,250,50)
    if st.button("Run backtest",type="primary"):
        if fast>=slow: st.error("Fast EMA must be smaller than Slow EMA."); return
        d=load_history(symbol,period,"1d")
        if d.empty or len(d)<slow+5: st.error("Insufficient historical data for this backtest."); return
        x=d[["Close"]].copy(); x["fast"]=x.Close.ewm(span=int(fast),adjust=False).mean(); x["slow"]=x.Close.ewm(span=int(slow),adjust=False).mean()
        x["signal"]=(x.fast>x.slow).astype(int); x["position"]=x.signal.shift(1).fillna(0)
        x["market_return"]=x.Close.pct_change().fillna(0); x["strategy_return"]=x.position*x.market_return
        x["equity"]=(1+x.strategy_return).cumprod(); x["buy_hold"]=(1+x.market_return).cumprod()
        total=(x.equity.iloc[-1]-1)*100; bh=(x.buy_hold.iloc[-1]-1)*100
        peak=x.equity.cummax(); dd=(x.equity/peak-1).min()*100
        trades=int((x.position.diff().fillna(x.position.iloc[0])!=0).sum())
        a,b,c,dcol=st.columns(4); a.metric("Strategy return",f"{total:+.2f}%"); b.metric("Buy & hold",f"{bh:+.2f}%"); c.metric("Max drawdown",f"{dd:.2f}%"); dcol.metric("Position changes",trades)
        st.line_chart(x[["equity","buy_hold"]]); st.dataframe(x.tail(15),use_container_width=True)


def render_alerts():
    st.subheader("🔔 Price Alert Checker")
    st.caption("Alerts are checked only when you press the button; this app does not run continuously in the background.")
    symbol=st.text_input("NSE symbol","RELIANCE",key="alert_symbol").upper().strip()
    level=st.number_input("Alert level (₹)",min_value=0.0,value=1000.0,step=1.0)
    direction=st.selectbox("Trigger when",["Price crosses above","Price crosses below"])
    if st.button("Check current/latest available price",type="primary"):
        d=load_history(symbol,"5d","1d")
        if d.empty: st.error("Could not retrieve price.")
        else:
            price=float(d.Close.iloc[-1]); st.metric(f"{symbol} latest available close",fmt_price(price))
            hit=price>=level if direction.endswith("above") else price<=level
            if hit: st.success(f"Condition met: {direction.lower()} ₹{level:,.2f}")
            else: st.info(f"Condition not met. Alert level: ₹{level:,.2f}")

with st.sidebar:
    st.title("🇮🇳 India NSE Trading Desk")
    st.caption("Yahoo Finance data · No FYERS · No broker execution")
    page=st.radio("Navigation",["Market Dashboard","NSE Scanner","Stock Analysis","Entry / Exit Planner","Position Sizing","Historical Backtest","Price Alerts"])
    st.divider()
    st.caption("NSE symbols use Yahoo Finance suffix .NS. Public data may be delayed, rate-limited or missing.")

st.title("🇮🇳 India NSE AI Trading Desk")
st.caption("Research and paper-trading dashboard adapted for Indian listed stocks. No live order placement.")
if page=="Market Dashboard": render_market_dashboard()
elif page=="NSE Scanner": render_scanner()
elif page=="Stock Analysis": render_analysis()
elif page=="Entry / Exit Planner": render_entry_exit()
elif page=="Position Sizing": render_position_size()
elif page=="Historical Backtest": render_backtest()
elif page=="Price Alerts": render_alerts()

st.divider()
st.caption("Disclaimer: Educational technical analysis only. Not investment advice. Verify prices with your broker/exchange before trading. This app does not provide a live NSE order book, F&O option chain, or guaranteed real-time prices.")
