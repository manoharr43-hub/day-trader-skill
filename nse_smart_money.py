"""NSE 18-agent technical scanner powered by Yahoo Finance (unofficial data).
The agents are transparent rule-based indicators, not autonomous AI or proof of institutional orders.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

NSE_EQUITY_DEFAULTS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "LT.NS", "BHARTIARTL.NS", "ITC.NS", "TATAMOTORS.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "ADANIPORTS.NS", "ONGC.NS",
]
NSE_FO_DEFAULTS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INFY.NS", "TCS.NS", "LT.NS", "BHARTIARTL.NS",
    "ITC.NS", "TATAMOTORS.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS",
    "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
]
INDEX_MAP = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "INDIA VIX": "^INDIAVIX"}
INTRADAY = {"5 min": ("5m", "5d"), "15 min": ("15m", "1mo"), "30 min": ("30m", "1mo"), "60 min": ("60m", "3mo")}
SWING = {"Daily": ("1d", "2y"), "Weekly": ("1wk", "5y")}


def clean_symbol(s: str) -> str:
    s = s.strip().upper()
    if not s:
        return ""
    if s.startswith("NSE:"):
        s = s[4:]
    s = re.sub(r"-(EQ|BE)$", "", s)
    if s.startswith("^") or s.endswith((".NS", ".BO")):
        return s
    if re.fullmatch(r"[A-Z0-9&_-]+", s):
        return s.replace("-", "") + ".NS"
    return s


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def _safe_last(series: pd.Series, default=np.nan):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def analyze_agents(symbol: str, df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> Dict:
    """Evaluate 18 independently explainable technical agents on completed bars."""
    if df is None or len(df) < 55:
        raise ValueError("At least 55 usable candles are required for the 18-agent analysis.")
    d = df.copy()
    c, h, l, o, v = (d[x].astype(float) for x in ("Close", "High", "Low", "Open", "Volume"))
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean() if len(c) >= 200 else c.rolling(min(50, len(c))).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    pdi = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    vol_avg = v.rolling(20).mean()
    rvol = v / vol_avg.replace(0, np.nan)
    typical = (h+l+c)/3
    vwap = (typical*v).cumsum() / v.replace(0, np.nan).cumsum()
    obv = (np.sign(c.diff()).fillna(0)*v).cumsum()
    obv_slope = obv.diff(5).iloc[-1] if len(obv) > 5 else 0
    last = float(c.iloc[-1]); prev = float(c.iloc[-2]); last_o=float(o.iloc[-1])
    last_h=float(h.iloc[-1]); last_l=float(l.iloc[-1])
    rsi_now = _safe_last(rsi, 50); macd_now=_safe_last(macd); macd_sig_now=_safe_last(macd_sig)
    adx_now=_safe_last(adx, 0); rvol_now=_safe_last(rvol, 1)
    atr_now=_safe_last(atr, 0); atr_pct=atr_now/last*100 if last else 0
    ema20_now=_safe_last(ema20); ema50_now=_safe_last(ema50); ema200_now=_safe_last(ema200)
    vwap_now=_safe_last(vwap, last)
    prev_high=float(h.iloc[-21:-1].max()); prev_low=float(l.iloc[-21:-1].min())
    recent_high=float(h.iloc[-6:-1].max()); recent_low=float(l.iloc[-6:-1].min())
    candle_range=max(last_h-last_l, 1e-9)
    close_location=(last-last_l)/candle_range
    body_pct=abs(last-last_o)/candle_range
    vol_multiple=rvol_now if np.isfinite(rvol_now) else 1.0
    signals: List[Tuple[str,str,str,int]] = []
    def add(name, direction, detail, weight=1):
        signals.append((name,direction,detail,weight))

    # 1 Market trend
    add("Market trend", "BUY" if last>ema20_now and ema20_now>ema50_now else "SELL" if last<ema20_now and ema20_now<ema50_now else "NEUTRAL", "Price/EMA20/EMA50 alignment")
    # 2 EMA structure
    add("EMA structure", "BUY" if ema20_now>ema50_now and last>ema200_now else "SELL" if ema20_now<ema50_now and last<ema200_now else "NEUTRAL", "EMA20/50/200 alignment; EMA200 uses 50-bar proxy if history is shorter")
    # 3 RSI
    add("RSI momentum", "BUY" if 52<=rsi_now<=70 else "SELL" if 30<=rsi_now<=48 else "NEUTRAL", f"RSI(14) {rsi_now:.1f}; extremes are not chased")
    # 4 MACD
    add("MACD momentum", "BUY" if macd_now>macd_sig_now else "SELL" if macd_now<macd_sig_now else "NEUTRAL", "MACD line versus signal")
    # 5 ADX directional trend
    add("ADX strength", "BUY" if adx_now>=20 and pdi.iloc[-1]>mdi.iloc[-1] else "SELL" if adx_now>=20 and pdi.iloc[-1]<mdi.iloc[-1] else "NEUTRAL", f"ADX {adx_now:.1f}; directional strength")
    # 6 Relative volume
    add("Relative volume", "BUY" if vol_multiple>=1.5 and last>=prev else "SELL" if vol_multiple>=1.5 and last<prev else "NEUTRAL", f"Current volume / 20-bar average = {vol_multiple:.2f}x")
    # 7 Volume expansion
    vol_trend = _safe_last(v.rolling(5).mean()) / max(_safe_last(v.rolling(20).mean(),1),1)
    add("Volume expansion", "BUY" if vol_trend>=1.25 and last>prev else "SELL" if vol_trend>=1.25 and last<prev else "NEUTRAL", f"5-bar avg / 20-bar avg = {vol_trend:.2f}x")
    # 8 VWAP
    add("VWAP position", "BUY" if last>vwap_now else "SELL" if last<vwap_now else "NEUTRAL", f"Close {last:.2f} vs cumulative VWAP {vwap_now:.2f}; VWAP is session-like approximation on fetched window")
    # 9 Candle pressure
    add("Candle pressure", "BUY" if close_location>=0.75 and last>last_o else "SELL" if close_location<=0.25 and last<last_o else "NEUTRAL", f"Close location in candle {close_location:.0%}; body/range {body_pct:.0%}")
    # 10 breakout
    add("Range breakout", "BUY" if last>prev_high else "SELL" if last<prev_low else "NEUTRAL", f"20-bar range high {prev_high:.2f}, low {prev_low:.2f}")
    # 11 BOS
    add("BOS structure", "BUY" if last>recent_high else "SELL" if last<recent_low else "NEUTRAL", "Close beyond previous 5-bar swing boundary")
    # 12 CHOCH
    prior_trend_up = c.iloc[-6] > c.iloc[-20] if len(c)>=20 else False
    choch_buy = (not prior_trend_up) and last>recent_high
    choch_sell = prior_trend_up and last<recent_low
    add("CHOCH shift", "BUY" if choch_buy else "SELL" if choch_sell else "NEUTRAL", "Simple swing-direction change proxy; not discretionary market structure")
    # 13 Liquidity sweep
    sweep_low = last_l < prev_low and last > prev_low
    sweep_high = last_h > prev_high and last < prev_high
    add("Liquidity sweep", "BUY" if sweep_low else "SELL" if sweep_high else "NEUTRAL", "Wick beyond 20-bar extreme followed by close back inside")
    # 14 FVG
    fvg_up = float(l.iloc[-1]) > float(h.iloc[-3])
    fvg_dn = float(h.iloc[-1]) < float(l.iloc[-3])
    add("Fair value gap", "BUY" if fvg_up else "SELL" if fvg_dn else "NEUTRAL", "3-candle imbalance proxy")
    # 15 Order block approximation
    ob_buy = (c.iloc[-2] < o.iloc[-2]) and (last > h.iloc[-2]) and vol_multiple>=1.2
    ob_sell = (c.iloc[-2] > o.iloc[-2]) and (last < l.iloc[-2]) and vol_multiple>=1.2
    add("Order block", "BUY" if ob_buy else "SELL" if ob_sell else "NEUTRAL", "Previous opposite candle displaced with volume; approximate zone, not visible order data")
    # 16 accumulation/distribution proxy
    mfm = ((c-l)-(h-c)) / (h-l).replace(0,np.nan)
    mfv = mfm.fillna(0)*v
    ad_line = mfv.cumsum()
    ad_delta = _safe_last(ad_line.diff(5),0)
    add("Accumulation proxy", "BUY" if ad_delta>0 and last>=prev else "SELL" if ad_delta<0 and last<=prev else "NEUTRAL", "Chaikin-style price/volume flow proxy; cannot identify institutions")
    # 17 relative strength vs NIFTY
    rel_dir="NEUTRAL"; rel_detail="Benchmark unavailable"
    if benchmark is not None and not benchmark.empty and "Close" in benchmark:
        b=benchmark["Close"].dropna()
        aligned=pd.concat([c.pct_change(10).rename("stock"), b.pct_change(10).rename("bench")],axis=1).dropna()
        if not aligned.empty:
            excess=float(aligned["stock"].iloc[-1]-aligned["bench"].iloc[-1])
            rel_dir="BUY" if excess>0.005 else "SELL" if excess<-0.005 else "NEUTRAL"
            rel_detail=f"10-bar excess return vs NIFTY {excess*100:.2f}%"
    add("Relative strength", rel_dir, rel_detail)
    # 18 volatility compression / expansion
    atr_series=atr/ c.replace(0,np.nan)
    atr_base=_safe_last(atr_series.rolling(20).mean(),atr_pct/100)
    atr_ratio=(atr_pct/100)/atr_base if atr_base else 1
    squeeze=atr_ratio<0.8
    expansion=atr_ratio>1.2
    add("Volatility regime", "BUY" if expansion and last>prev else "SELL" if expansion and last<prev else "NEUTRAL", f"ATR% {atr_pct:.2f}; ratio to 20-bar ATR% avg {atr_ratio:.2f}; squeeze={squeeze}")

    buys=sum(1 for _,d,_,_ in signals if d=="BUY")
    sells=sum(1 for _,d,_,_ in signals if d=="SELL")
    neutral=18-buys-sells
    net=buys-sells
    if buys>=12 and buys-sells>=6: verdict="🟢 BULLISH CONFLUENCE"
    elif sells>=12 and sells-buys>=6: verdict="🔴 BEARISH CONFLUENCE"
    elif buys>=8 and buys>sells: verdict="🟡 BUY WATCH"
    elif sells>=8 and sells>buys: verdict="🟡 SELL WATCH"
    else: verdict="⚪ NO CLEAR CONFLUENCE"
    # score is a rule count, deliberately not a probability
    return {
        "symbol":symbol,"price":last,"change_pct":(last/prev-1)*100 if prev else 0,
        "buy_agents":buys,"sell_agents":sells,"neutral_agents":neutral,"net_score":net,
        "verdict":verdict,"rsi":rsi_now,"rvol":vol_multiple,"adx":adx_now,
        "atr_pct":atr_pct,"ema20":ema20_now,"ema50":ema50_now,"ema200":ema200_now,
        "vwap":vwap_now,"breakout_level":prev_high,"breakdown_level":prev_low,
        "squeeze":squeeze,"signals":signals,"frame":d,
    }


def render_smart_money_page():
    st.title("🇮🇳 NSE 18-Agent Smart Money & Big Move Scanner")
    st.caption("Yahoo Finance data · No FYERS · Intraday + Swing · Equity + F&O watchlists")
    st.warning("Yahoo Finance is unofficial and may be delayed/rate-limited. These 18 agents are transparent technical rules, not proof of institutional orders or guaranteed predictions. No live order-book depth or option-chain feed is provided.")

    c1,c2,c3=st.columns(3)
    with c1:
        market=st.selectbox("Market universe",["Both: Equity + F&O","NSE Equity watchlist","F&O watchlist"],key="sm_market")
    with c2:
        timeframe=st.selectbox("Timeframe",["Both: Intraday + Swing","Intraday","Swing"],key="sm_tf")
    with c3:
        benchmark_name=st.selectbox("Benchmark",list(INDEX_MAP),key="sm_benchmark")
    defaults = list(dict.fromkeys(NSE_EQUITY_DEFAULTS+NSE_FO_DEFAULTS)) if market.startswith("Both") else (NSE_EQUITY_DEFAULTS if market.startswith("NSE Equity") else NSE_FO_DEFAULTS)
    symbols_text=st.text_area("NSE symbols (comma separated; editable)",", ".join(defaults),height=90,key="sm_symbols")
    symbols=list(dict.fromkeys(clean_symbol(x) for x in symbols_text.replace("\n",",").split(",") if x.strip()))
    symbols=[s for s in symbols if s]
    max_symbols=st.slider("Maximum stocks per run",5,60,min(30,max(5,len(symbols))),5,key="sm_max")
    symbols=symbols[:max_symbols]

    benchmark_symbol=INDEX_MAP[benchmark_name]
    run=st.button("🔎 RUN 18 AGENT SCAN",type="primary",use_container_width=True)
    if run:
        st.session_state["sm_results"]={}
        modes=[]
        if timeframe in ("Both: Intraday + Swing","Intraday"):
            modes += [("Intraday · "+name,iv,per) for name,(iv,per) in INTRADAY.items() if name in ("15 min","60 min")]
        if timeframe in ("Both: Intraday + Swing","Swing"):
            modes += [("Swing · "+name,iv,per) for name,(iv,per) in SWING.items()]
        progress=st.progress(0.0); status=st.empty(); results={}
        total=max(len(symbols)*len(modes),1); done=0
        for mode_name,interval,period in modes:
            try:
                bench=fetch_history(benchmark_symbol,interval,period)
            except Exception:
                bench=pd.DataFrame()
            for sym in symbols:
                done+=1; status.write(f"Analyzing {sym} — {mode_name} ({done}/{total})")
                try:
                    hist=fetch_history(sym,interval,period)
                    if hist.empty:
                        continue
                    result=analyze_agents(sym,hist,bench)
                    result["mode"]=mode_name
                    results[(mode_name,sym)]=result
                except Exception as exc:
                    results[(mode_name,sym)]={"error":str(exc),"mode":mode_name,"symbol":sym}
                progress.progress(done/total)
        st.session_state["sm_results"]=results
        status.success(f"Scan completed: {sum(1 for x in results.values() if 'error' not in x)} usable analyses.")

    results=st.session_state.get("sm_results",{})
    if results:
        rows=[]
        for (_,sym),r in results.items():
            if "error" in r: continue
            rows.append({"Timeframe":r["mode"],"Stock":sym,"LTP (last Yahoo close)":round(r["price"],2),"Change %":round(r["change_pct"],2),"BUY agents":r["buy_agents"],"SELL agents":r["sell_agents"],"Neutral":r["neutral_agents"],"Net rule score":r["net_score"],"RVOL":round(r["rvol"],2),"RSI":round(r["rsi"],1),"Signal":r["verdict"]})
        if rows:
            table=pd.DataFrame(rows).sort_values(["BUY agents","Net rule score"],ascending=False)
            st.subheader("Scanner results — rule count, not probability")
            st.dataframe(table,use_container_width=True,hide_index=True)
            st.download_button("Download scan CSV",table.to_csv(index=False).encode("utf-8"),"nse_18_agent_scan.csv","text/csv")
            choices=[f"{r['Timeframe']} | {r['Stock']}" for r in rows]
            selected=st.selectbox("Detailed 18-agent chart analysis",choices,key="sm_detail")
            mode_name,sym=selected.split(" | ",1)
            detail=results.get((mode_name,sym))
            if detail and "error" not in detail:
                a,b,c,d=st.columns(4)
                a.metric("BUY agents",f"{detail['buy_agents']}/18")
                b.metric("SELL agents",f"{detail['sell_agents']}/18")
                c.metric("RVOL",f"{detail['rvol']:.2f}x")
                d.metric("RSI",f"{detail['rsi']:.1f}")
                st.subheader(detail["verdict"])
                fig=go.Figure(data=[go.Candlestick(x=detail["frame"].index,open=detail["frame"]["Open"],high=detail["frame"]["High"],low=detail["frame"]["Low"],close=detail["frame"]["Close"],name=sym)])
                st.plotly_chart(fig,use_container_width=True)
                st.write(f"**Breakout watch level:** ₹{detail['breakout_level']:.2f} · **Breakdown watch level:** ₹{detail['breakdown_level']:.2f}")
                st.write(f"**EMA20:** ₹{detail['ema20']:.2f} · **EMA50:** ₹{detail['ema50']:.2f} · **EMA200:** ₹{detail['ema200']:.2f} · **VWAP approximation:** ₹{detail['vwap']:.2f}")
                agent_df=pd.DataFrame(detail["signals"],columns=["Agent","Direction","Reason","Weight"])
                st.dataframe(agent_df.drop(columns=["Weight"]),use_container_width=True,hide_index=True)
                st.caption("Order block, FVG, BOS/CHOCH and liquidity sweep are simplified OHLCV proxies. They are not actual exchange order-book observations.")
        errors=[r for r in results.values() if "error" in r]
        if errors:
            with st.expander(f"Skipped / failed symbols ({len(errors)})"):
                st.dataframe(pd.DataFrame(errors),use_container_width=True,hide_index=True)
    else:
        st.info("Enter or keep the NSE symbols and click RUN 18 AGENT SCAN. For wider coverage, paste more valid Yahoo NSE tickers in the symbol box.")

    with st.expander("18 agents — what each one checks"):
        st.markdown("""
1. Market trend · 2. EMA structure · 3. RSI momentum · 4. MACD momentum · 5. ADX strength · 6. Relative volume · 7. Volume expansion · 8. VWAP position · 9. Candle pressure · 10. Range breakout · 11. BOS structure · 12. CHOCH shift · 13. Liquidity sweep · 14. Fair Value Gap · 15. Order block approximation · 16. Accumulation/distribution proxy · 17. Relative strength vs NIFTY · 18. Volatility regime.

**How to read it:** BUY/SELL counts are the number of rules aligned in that direction. They are not win rates, confidence percentages, or probabilities. A high score must be validated with out-of-sample historical backtesting before any probability claim.
""")
