"""
NSE AI PRO V17 - Yahoo Finance edition
--------------------------------------
FYERS authentication and fyers_apiv3 have been removed.

The YahooFinanceClient intentionally exposes a small FYERS-shaped history()
and quotes() interface so existing page modules that use those read methods
can continue to work with minimal/no changes.

IMPORTANT LIMITATIONS:
- Yahoo Finance is unofficial and may be delayed, rate-limited, or incomplete.
- Yahoo Finance does not provide a dependable NSE real-time feed.
- This adapter does not place broker orders or provide a reliable Indian
  derivatives/options chain. Broker execution must remain a separate,
  explicitly configured integration.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception as exc:
    st.error(
        "Yahoo Finance package is missing. Install it with: "
        "pip install yfinance"
    )
    st.exception(exc)
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NSE AI PRO V17",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("nse_ai_pro")

# ---------------- SYMBOL / INTERVAL HELPERS ----------------
INDEX_SYMBOLS = {
    "NSE:NIFTY50-INDEX": "^NSEI",
    "NSE:NIFTYBANK-INDEX": "^NSEBANK",
    "NSE:FINNIFTY-INDEX": "^CNXFINANCE",
    "NSE:INDIAVIX-INDEX": "^INDIAVIX",
    "BSE:SENSEX-INDEX": "^BSESN",
}
RESOLUTION_MAP = {
    "1": "1m",
    "2": "2m",
    "5": "5m",
    "10": "15m",   # Yahoo has no 10-minute interval; use 15m approximation.
    "15": "15m",
    "30": "30m",
    "60": "60m",
    "D": "1d",
    "1D": "1d",
    "W": "1wk",
    "M": "1mo",
}


def normalize_symbol(symbol: str) -> str:
    """Convert common FYERS/NSE symbols to Yahoo Finance tickers."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        raise ValueError("Symbol is empty.")

    if raw in INDEX_SYMBOLS:
        return INDEX_SYMBOLS[raw]

    # FYERS format: NSE:RELIANCE-EQ -> RELIANCE.NS
    match = re.match(r"^(?:NSE|BSE):([A-Z0-9&_-]+?)(?:-EQ|-BE)?$", raw)
    if match:
        name = match.group(1).replace("-", "")
        suffix = ".BO" if raw.startswith("BSE:") else ".NS"
        return name + suffix

    if raw.endswith("-EQ"):
        raw = raw[:-3] + ".NS"
    elif raw.endswith("-BE"):
        raw = raw[:-3] + ".NS"
    elif raw.startswith("^"):
        return raw
    elif "." not in raw and raw.isalnum():
        raw += ".NS"
    return raw


class YahooFinanceClient:
    """
    Read-only Yahoo Finance adapter.

    Compatible methods:
      client.history(data={symbol,resolution,range_from,range_to,...})
      client.quotes(data={"symbols": "NSE:RELIANCE-EQ,NSE:TCS-EQ"})

    Returned history format mirrors the FYERS candle response to reduce
    changes needed in existing scanner/page modules.
    """

    def __init__(self):
        self.provider = "Yahoo Finance (unofficial)"
        self.last_warnings: list[str] = []

    @staticmethod
    def _interval(resolution: Any) -> str:
        key = str(resolution or "D")
        if key not in RESOLUTION_MAP:
            raise ValueError(
                f"Unsupported timeframe '{key}'. Use 1, 5, 15, 30, 60, or D."
            )
        return RESOLUTION_MAP[key]

    @staticmethod
    def _limit_intraday_days(interval: str, requested_days: int) -> int:
        # Yahoo chart endpoint has strict lookback limits for intraday bars.
        limits = {
            "1m": 7,
            "2m": 60,
            "5m": 60,
            "15m": 60,
            "30m": 60,
            "60m": 730,
        }
        return min(max(int(requested_days), 1), limits.get(interval, 36500))

    def history(self, data: dict | None = None, **kwargs) -> dict:
        payload = dict(data or {})
        payload.update(kwargs)

        symbol = normalize_symbol(payload.get("symbol", ""))
        interval = self._interval(payload.get("resolution", "D"))

        end_raw = payload.get("range_to")
        start_raw = payload.get("range_from")
        end_date = (
            dt.datetime.strptime(str(end_raw)[:10], "%Y-%m-%d").date()
            if end_raw else dt.date.today() + dt.timedelta(days=1)
        )
        if start_raw:
            start_date = dt.datetime.strptime(
                str(start_raw)[:10], "%Y-%m-%d"
            ).date()
            requested_days = max((end_date - start_date).days, 1)
        else:
            requested_days = 30
            start_date = end_date - dt.timedelta(days=requested_days)

        self.last_warnings = []
        if interval.endswith("m") or interval.endswith("h"):
            allowed_days = self._limit_intraday_days(interval, requested_days)
            if allowed_days < requested_days:
                start_date = end_date - dt.timedelta(days=allowed_days)
                self.last_warnings.append(
                    f"Yahoo Finance limits {interval} history to about "
                    f"{allowed_days} days; requested range was shortened."
                )

        try:
            frame = yf.download(
                tickers=symbol,
                start=start_date.isoformat(),
                end=(end_date + dt.timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=False,
                prepost=False,
                progress=False,
                threads=False,
                group_by="column",
                timeout=20,
            )
        except Exception as exc:
            return {"s": "error", "code": -1, "message": str(exc), "candles": []}

        if frame is None or frame.empty:
            return {
                "s": "error",
                "code": -1,
                "message": (
                    f"No Yahoo Finance candles returned for {symbol} "
                    f"({interval}). Check ticker, market session, and lookback."
                ),
                "candles": [],
            }

        # yfinance may return MultiIndex columns even for a single ticker.
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        frame = frame.rename(columns={c: str(c).title() for c in frame.columns})
        required = ["Open", "High", "Low", "Close"]
        if not all(col in frame.columns for col in required):
            return {
                "s": "error",
                "code": -1,
                "message": f"Unexpected Yahoo Finance columns: {list(frame.columns)}",
                "candles": [],
            }

        if "Volume" not in frame.columns:
            frame["Volume"] = 0

        candles = []
        for timestamp, row in frame.iterrows():
            values = [row.get(c, np.nan) for c in required]
            if any(pd.isna(v) for v in values):
                continue
            stamp = pd.Timestamp(timestamp)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("Asia/Kolkata")
            unix_seconds = int(stamp.timestamp())
            volume = row.get("Volume", 0)
            candles.append([
                unix_seconds,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                float(volume) if pd.notna(volume) else 0.0,
            ])

        if not candles:
            return {
                "s": "error",
                "code": -1,
                "message": f"No valid OHLC candles for {symbol}.",
                "candles": [],
            }

        return {
            "s": "ok",
            "code": 200,
            "message": "Success",
            "candles": candles,
            "symbol": symbol,
            "interval": interval,
            "warnings": self.last_warnings.copy(),
        }

    def quotes(self, data: dict | None = None, **kwargs) -> dict:
        payload = dict(data or {})
        payload.update(kwargs)
        raw_symbols = payload.get("symbols") or payload.get("symbol") or ""
        symbols = [
            item.strip() for item in str(raw_symbols).split(",") if item.strip()
        ]
        result = []

        for original in symbols:
            ticker = normalize_symbol(original)
            try:
                info = yf.Ticker(ticker)
                fast = getattr(info, "fast_info", {}) or {}
                price = fast.get("lastPrice")
                previous = fast.get("previousClose")
                if price is None:
                    hist = info.history(period="5d", interval="1d")
                    if hist is not None and not hist.empty:
                        price = float(hist["Close"].dropna().iloc[-1])
                        previous = (
                            float(hist["Close"].dropna().iloc[-2])
                            if len(hist["Close"].dropna()) > 1 else price
                        )
                if price is None:
                    continue
                price = float(price)
                previous = float(previous) if previous is not None else price
                result.append({
                    "n": original,
                    "s": "ok",
                    "v": {
                        "lp": price,
                        "prev_close": previous,
                        "ch": price - previous,
                        "chp": ((price - previous) / previous * 100)
                               if previous else 0.0,
                        "provider": self.provider,
                    },
                })
            except Exception as exc:
                LOGGER.warning("Quote failed for %s: %s", original, exc)

        return {"s": "ok" if result else "error", "d": result}

    def market_status(self):
        return {
            "s": "ok",
            "market_status": "Yahoo Finance availability varies by symbol/session",
        }


# ---------------- SESSION ----------------
st.session_state.setdefault("main_navigation", "🏠 Dashboard")
st.session_state.setdefault("bt_result", None)
st.session_state.setdefault("yahoo_client", None)

client = st.session_state.get("yahoo_client")
if client is None:
    client = YahooFinanceClient()
    st.session_state["yahoo_client"] = client

# ---------------- AI BACKTEST ----------------
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()


def adx(frame: pd.DataFrame, length: int = 14) -> pd.Series:
    up = frame["High"].diff()
    down = -frame["Low"].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=frame.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=frame.index
    )
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - frame["High"].shift(1)).abs(),
            (frame["Low"] - frame["Low"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    average_range = true_range.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / average_range
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / length, adjust=False, min_periods=length
    ).mean() / average_range
    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    output = pd.DataFrame(index=frame.index)
    output["ema20"] = ema(close, 20)
    output["ema50"] = ema(close, 50)
    output["rsi"] = rsi(close)
    macd_fast, macd_slow = ema(close, 12), ema(close, 26)
    macd_line = macd_fast - macd_slow
    output["macd_hist"] = macd_line - macd_line.ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    output["atr"] = atr(frame)
    output["adx"] = adx(frame)
    average_volume = volume.rolling(20, min_periods=20).mean()
    output["rel_vol"] = volume / average_volume.replace(0, np.nan)
    output["ret1"] = close.pct_change()
    future_close = close.shift(-1)
    output["target"] = (future_close > close).astype(float)
    output.loc[future_close.isna(), "target"] = np.nan
    return output.replace([np.inf, -np.inf], np.nan).dropna()


def make_model():
    return xgb.XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )


def get_history_frame(data_client, symbol: str, days: int, resolution: str):
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    response = data_client.history(data={
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": end.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    })
    if not isinstance(response, dict) or response.get("s") != "ok":
        raise RuntimeError(response.get("message", "Yahoo history request failed"))
    candles = response.get("candles", [])
    if not candles:
        raise RuntimeError("No historical candles returned.")
    frame = pd.DataFrame(
        candles,
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], unit="s")
    frame = frame.set_index("Timestamp")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Open", "High", "Low", "Close"])


def run_backtest(data_client, symbol, resolution, days, test_bars):
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost missing. Add xgboost>=2.1.0 to requirements.txt.")
    data = features(get_history_frame(data_client, symbol, days, resolution))
    if len(data) < 140:
        raise RuntimeError(f"Only {len(data)} usable rows; need at least 140.")
    test_bars = min(test_bars, len(data) - 100)
    start = len(data) - test_bars
    feature_columns = [column for column in data.columns if column != "target"]
    actual, predicted, probabilities = [], [], []

    for index in range(start, len(data)):
        train = data.iloc[:index]
        test = data.iloc[[index]]
        if train["target"].nunique() < 2:
            continue
        model = make_model()
        model.fit(train[feature_columns], train["target"])
        probability = float(model.predict_proba(test[feature_columns])[0, 1])
        actual.append(int(test["target"].iloc[0]))
        predicted.append(int(probability >= 0.5))
        probabilities.append(probability)

    if not actual:
        raise RuntimeError("No valid walk-forward samples.")

    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "Symbol": symbol,
        "Resolution": resolution,
        "History Days Requested": days,
        "Accuracy %": round(100 * (tp + tn) / len(y_true), 2),
        "Precision UP %": round(100 * tp / max(tp + fp, 1), 2),
        "Recall UP %": round(100 * tp / max(tp + fn, 1), 2),
        "Average P(UP) %": round(100 * float(np.mean(probabilities)), 2),
        "Test Samples": len(y_true),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def ai_backtest_ui(data_client):
    st.divider()
    with st.expander("🧪 AI BACKTEST — Historical Walk-Forward Accuracy", expanded=False):
        st.caption(
            "Historical accuracy is not a guarantee of future performance. "
            "Yahoo Finance data may be delayed or incomplete."
        )
        if not XGB_AVAILABLE:
            st.warning("XGBoost is not installed. Add xgboost>=2.1.0 to requirements.txt.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox(
                "Select stock / index",
                [
                    "NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ",
                    "NSE:INFY-EQ", "NSE:SBIN-EQ", "NSE:ICICIBANK-EQ",
                    "NSE:NIFTYBANK-INDEX", "NSE:NIFTY50-INDEX",
                ],
                key="bt_symbol",
            )
        with col2:
            resolution = st.selectbox(
                "Timeframe", ["15", "30", "60", "D"], index=2,
                key="bt_resolution",
            )
        with col3:
            bars = st.slider("Test bars", 20, 100, 60, 10, key="bt_bars")
        days = st.slider("Historical days", 90, 500, 180, 30, key="bt_days")

        if st.button(
            "🚀 RUN AI BACKTEST", type="primary",
            use_container_width=True, key="run_bt",
        ):
            with st.spinner("Downloading Yahoo Finance data and running backtest..."):
                try:
                    st.session_state["bt_result"] = run_backtest(
                        data_client, symbol, resolution, days, bars
                    )
                except Exception as exc:
                    st.session_state["bt_result"] = None
                    st.error(f"Backtest failed: {exc}")

        result = st.session_state.get("bt_result")
        if result:
            a, b, c, d = st.columns(4)
            a.metric("Actual Accuracy", f"{result['Accuracy %']:.2f}%")
            b.metric("Precision UP", f"{result['Precision UP %']:.2f}%")
            c.metric("Recall UP", f"{result['Recall UP %']:.2f}%")
            d.metric("Test Samples", result["Test Samples"])
            result_frame = pd.DataFrame([result])
            st.dataframe(result_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Export Backtest CSV",
                result_frame.to_csv(index=False).encode("utf-8"),
                f"ai_backtest_{dt.date.today()}.csv",
                "text/csv",
                key="bt_csv",
            )


# ---------------- PAGE ROUTING ----------------
PAGE_MODULES = {
    "🏠 Dashboard": ("dashboard", "show_dashboard"),
    "📈 Market": ("market", "show_market"),
    "🧠 AI Market Intelligence": ("ai_market_intelligence", "show_ai_market_intelligence"),
    "💼 Portfolio": ("portfolio", "show_portfolio"),
    "📋 Orders": ("orders", "show_orders"),
    "⚙️ Option Chain": ("option_chain", "show_option_chain"),
    "🤖 Scanner": ("scanner", "show_scanner"),
    "📊 Charts": ("charts", "show_charts"),
    "💹 Trading": ("trading", "show_trading"),
    "👤 Profile": ("profile", "show_profile"),
    "⚙️ Settings": ("settings", "show_settings"),
    "🤖 AI Chart Analysis": ("ai_chart_analysis", "show_ai_chart_analysis"),
    "🤖 AI Multi-Agent Scanner": ("ai_multi_agent_scanner", "show_ai_multi_agent_scanner"),
}


def render_page(menu: str, data_client):
    if menu == "🧠 AI Market Intelligence":
        module_name, function_name = PAGE_MODULES[menu]
        try:
            module = importlib.import_module(module_name)
            getattr(module, function_name)(data_client)
            ai_backtest_ui(data_client)
        except ModuleNotFoundError as exc:
            st.error(f"Page module not found: {exc.name}. Keep the existing project module in the app folder.")
        except Exception as exc:
            st.error(f"Error loading {menu}: {exc}")
            st.exception(exc)
        return

    module_info = PAGE_MODULES.get(menu)
    if not module_info:
        st.warning("Unknown page selection.")
        return

    module_name, function_name = module_info
    try:
        module = importlib.import_module(module_name)
        page_function = getattr(module, function_name)
        page_function(data_client)
    except ModuleNotFoundError as exc:
        st.error(
            f"Required page module '{exc.name}' was not found. "
            "This app.py expects your existing page .py files to remain beside app.py."
        )
    except Exception as exc:
        st.error(f"❌ Error loading {menu}: {exc}")
        st.exception(exc)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📈 NSE AI PRO V17")
    st.caption("Data source: Yahoo Finance · No FYERS login")
    items = list(PAGE_MODULES.keys())
    current = st.session_state.get("main_navigation", items[0])
    if current not in items:
        current = items[0]
    st.radio(
        "Navigation",
        items,
        index=items.index(current),
        key="main_navigation",
    )
    st.divider()
    st.caption("Main entry point: app.py")
    st.caption("Yahoo Finance is unofficial and may be delayed.")
    if st.button("🔄 Clear app cache / session", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in {"main_navigation"}:
                del st.session_state[key]
        st.rerun()

# ---------------- DATA STATUS ----------------
with st.sidebar:
    with st.expander("Data source status"):
        st.write("Provider: Yahoo Finance")
        st.write("Authentication: Not required")
        st.write("Broker orders: Not supported by this data adapter")
        st.write("Options chain: Requires a separate supported source")

# ---------------- RUN SELECTED PAGE ----------------
render_page(st.session_state["main_navigation"], client)
