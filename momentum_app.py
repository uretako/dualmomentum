import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="Dual Momentum Analyzer", page_icon="📈", layout="wide")
st.title("📈 Crypto + Traditional Assets Dual Momentum")
st.caption("Select any date — past or present — to run the momentum strategy as of that day.")

with st.sidebar:
    st.header("⚙️ Settings")
    analysis_date = st.date_input("Analysis Date", value=date.today(), max_value=date.today())
    lookback_months = st.selectbox("Lookback Period (months)", options=[1, 3, 6, 12, 24], index=3)
    custom_tickers_input = st.text_input("Add Custom Tickers (comma-separated)", placeholder="e.g., AAPL, MSFT, QQQ")
    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Default Universe**")
    st.markdown("""
- BTC-USD — Bitcoin
- ETH-USD — Ethereum
- SOL-USD — Solana
- BNB-USD — BNB
- ADA-USD — Cardano
- SPY — S&P 500 ETF
- GLD — Gold ETF
- IYR — Real Estate ETF
- TLT — Long-Term Treasuries
- BIL — Cash / Safety
    """)


def download_close(sym, start, end):
    """Robustly download Close prices for any yfinance version."""
    try:
        raw = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        close = raw["Close"]
        # yfinance 1.x with multi-ticker returns MultiIndex — flatten
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        # Force to plain Series with DatetimeIndex
        s = pd.Series(close.values.flatten(), index=pd.to_datetime(raw.index), name=sym)
        s = s.dropna()
        return s if len(s) > 0 else None
    except Exception:
        return None


def get_monthly_prices(symbols, analysis_date, lookback_months):
    start = analysis_date - timedelta(days=365 * 5 + 60)
    end = analysis_date + timedelta(days=5)

    series_list = []
    failed = []

    progress = st.progress(0, text="Downloading price data...")
    for i, sym in enumerate(symbols):
        s = download_close(sym, start, end)
        if s is not None:
            series_list.append(s.rename(sym))
        else:
            failed.append(sym)
        progress.progress((i + 1) / len(symbols), text=f"Downloading {sym}...")
    progress.empty()

    if failed:
        st.warning(f"Could not download: {', '.join(failed)}")

    if len(series_list) < 2:
        st.error("Not enough assets downloaded successfully.")
        return None

    # Safe concat — aligns on date index automatically
    prices = pd.concat(series_list, axis=1)
    prices = prices[prices.index <= pd.Timestamp(analysis_date)]
    prices = prices.ffill()

    monthly = prices.resample("ME").last()
    min_rows = lookback_months + 4
    monthly = monthly.dropna(thresh=min_rows, axis=1)

    if len(monthly) < lookback_months + 1:
        st.error("Not enough historical data for selected lookback.")
        return None

    return monthly


def calc_return(monthly, months_back):
    if len(monthly) < months_back + 1:
        return pd.Series(np.nan, index=monthly.columns)
    recent = monthly.iloc[-1]
    base = monthly.iloc[-(months_back + 1)]
    return ((recent / base) - 1) * 100


def calc_risk_metrics(monthly, lookback_months):
    returns = monthly.pct_change().dropna()
    if len(returns) >= lookback_months:
        returns = returns.tail(lookback_months)
    metrics = {}
    for col in returns.columns:
        r = returns[col].dropna()
        if len(r) > 1:
            vol = r.std() * np.sqrt(12) * 100
            sharpe = (r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else 0
            cum = (1 + r).cumprod()
            max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
            worst = r.min() * 100
            metrics[col] = {
                "Volatility %": round(vol, 2),
                "Max Drawdown %": round(max_dd, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Worst Month %": round(worst, 2)
            }
    return pd.DataFrame(metrics).T


def calc_forward_returns(symbol, analysis_date):
    end = min(analysis_date + timedelta(days=100), date.today())
    if end <= analysis_date:
        return None, None
    s = download_close(symbol, analysis_date, end + timedelta(days=2))
    if s is None or len(s) < 2:
        return None, None
    base = s.iloc[0]
    ret_1m = round(((s.iloc[min(21, len(s)-1)] / base) - 1) * 100, 2) if len(s) >= 15 else None
    ret_3m = round(((s.iloc[min(63, len(s)-1)] / base) - 1) * 100, 2) if len(s) >= 45 else None
    return ret_1m, ret_3m


if run:
    CASH = "BIL"
    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD",
                       "SPY", "GLD", "IYR", "TLT", CASH]
    custom = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()] if custom_tickers_input.strip() else []
    all_symbols = list(dict.fromkeys(default_symbols + custom))
    is_past = analysis_date < date.today()

    monthly = get_monthly_prices(all_symbols, analysis_date, lookback_months)

    if monthly is not None:
        mom = calc_return(monthly, lookback_months)
        ret_1m = calc_return(monthly, 1)
        ret_3m = calc_return(monthly, 3)
        risk = calc_risk_metrics(monthly, lookback_months)

        bil_mom = mom.get(CASH, 0.0)
        if pd.isna(bil_mom):
            bil_mom = 0.0

        results = pd.DataFrame({
            "1M Return %": ret_1m,
            "3M Return %": ret_3m,
            f"{lookback_months}M Momentum %": mom,
            "Beats Cash?": mom.apply(lambda x: "✅ Yes" if pd.notna(x) and x > bil_mom else "❌ No"),
        }).join(risk).sort_values(f"{lookback_months}M Momentum %", ascending=False)

        investable = [c for c in monthly.columns if c != CASH]
        mom_investable = mom[investable]
        best = mom_investable.idxmax()
        best_mom = mom_investable.max()

        # Antonacci Dual Momentum — two rules:
        # 1. RELATIVE: pick asset with highest lookback return among investable assets
        # 2. ABSOLUTE: that asset must also beat BIL (cash) over same lookback
        #    If it fails absolute test, go to cash regardless of relative ranking
        passes_relative = pd.notna(best_mom)
        passes_absolute = passes_relative and (best_mom > bil_mom)

        if passes_absolute:
            recommendation = best
            reason = (f"highest relative momentum (+{best_mom:.2f}%) "
                      f"and beats cash (BIL: {bil_mom:.2f}%) — Antonacci dual momentum signal")
        elif passes_relative and not passes_absolute:
            recommendation = CASH
            reason = (f"{best} had highest momentum (+{best_mom:.2f}%) "
                      f"but failed absolute test — does not beat cash (BIL: {bil_mom:.2f}%)")
        else:
            recommendation = CASH
            reason = "no assets have sufficient data for momentum calculation"

        c1, c2, c3 = st.columns(3)
        c1.metric("📅 Analysis Date", str(analysis_date))
        c2.metric("🏆 Recommended Asset", recommendation)
        c3.metric("📊 Momentum", f"{best_mom:.2f}%" if pd.notna(best_mom) else "N/A")
        st.info(f"**Strategy says:** Invest in **{recommendation}** — {reason}")

        if is_past:
            st.markdown("---")
            st.subheader("⏩ Forward Returns from Analysis Date")
            st.caption("What actually happened after the strategy signal.")
            with st.spinner(f"Fetching forward data for {recommendation}..."):
                fwd_1m, fwd_3m = calc_forward_returns(recommendation, analysis_date)
            fc1, fc2 = st.columns(2)
            fc1.metric(f"1M Forward ({recommendation})", f"{fwd_1m:+.2f}%" if fwd_1m is not None else "N/A")
            fc2.metric(f"3M Forward ({recommendation})", f"{fwd_3m:+.2f}%" if fwd_3m is not None else "N/A")

        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 Momentum Strategy", "⚠️ Risk Analysis"])

        with tab1:
            st.subheader("Performance & Risk Matrix")

            def color_val(val):
                if pd.isna(val): return ""
                return "color: green; font-weight: bold" if val > 0 else "color: red; font-weight: bold"

            def color_dd(val):
                if pd.isna(val): return ""
                if val < -20: return "background-color: #FF6B6B"
                if val < -10: return "background-color: #FFD700"
                return "background-color: #90EE90"

            styled = (results.style
                      .map(color_val, subset=[f"{lookback_months}M Momentum %", "1M Return %", "3M Return %"])
                      .map(color_dd, subset=["Max Drawdown %"])
                      .format("{:.2f}", na_rep="N/A"))
            st.dataframe(styled, use_container_width=True)

            mom_df = results[[f"{lookback_months}M Momentum %"]].reset_index()
            mom_df.columns = ["Asset", "Momentum %"]
            mom_df["Color"] = mom_df["Momentum %"].apply(lambda x: "Positive" if x > 0 else "Negative")
            fig = px.bar(mom_df, x="Asset", y="Momentum %", color="Color",
                         color_discrete_map={"Positive": "#90EE90", "Negative": "#FF6B6B"},
                         title=f"{lookback_months}-Month Momentum by Asset")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            ca, cb = st.columns(2)
            with ca:
                vol_df = results[["Volatility %"]].reset_index()
                vol_df.columns = ["Asset", "Volatility %"]
                st.plotly_chart(px.bar(vol_df, x="Asset", y="Volatility %",
                                       color="Volatility %", color_continuous_scale="Reds",
                                       title="Annualized Volatility"), use_container_width=True)
            with cb:
                dd_df = results[["Max Drawdown %"]].reset_index()
                dd_df.columns = ["Asset", "Max Drawdown %"]
                st.plotly_chart(px.bar(dd_df, x="Asset", y="Max Drawdown %",
                                       color="Max Drawdown %", color_continuous_scale="Blues_r",
                                       title="Maximum Drawdown"), use_container_width=True)

            scatter_df = results[[f"{lookback_months}M Momentum %", "Volatility %"]].reset_index().dropna()
            scatter_df.columns = ["Asset", "Momentum %", "Volatility %"]
            fig4 = px.scatter(scatter_df, x="Volatility %", y="Momentum %",
                              text="Asset", color="Asset",
                              title="Risk vs Return — ideal assets are top-left")
            fig4.update_traces(textposition="top center", marker_size=12)
            st.plotly_chart(fig4, use_container_width=True)

        st.caption(f"Data via Yahoo Finance. Analysis as of {analysis_date}. Not financial advice.")
