import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Dual Momentum Analyzer", page_icon="📈", layout="wide")

st.title("📈 Crypto + Traditional Assets Dual Momentum")
st.caption("Select any date — past or present — to run the momentum strategy as of that day.")

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")

    analysis_date = st.date_input(
        "Analysis Date",
        value=date.today(),
        max_value=date.today()
    )

    lookback_months = st.slider(
        "Lookback Period (months)",
        min_value=1, max_value=60, value=12
    )

    custom_tickers_input = st.text_input(
        "Add Custom Tickers (comma-separated)",
        placeholder="e.g., AAPL, MSFT, QQQ"
    )

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

# ---- Core Logic ----
def get_monthly_prices(symbols, analysis_date, lookback_months):
    start = analysis_date - timedelta(days=365 * 5 + 31)
    end = analysis_date + timedelta(days=2)

    px_list = {}
    failed = []

    progress = st.progress(0, text="Downloading price data...")
    for i, sym in enumerate(symbols):
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
            if not df.empty:
                px_list[sym] = df["Close"]
        except Exception:
            failed.append(sym)
        progress.progress((i + 1) / len(symbols), text=f"Downloading {sym}...")

    progress.empty()

    if failed:
        st.warning(f"Could not download: {', '.join(failed)}")

    if len(px_list) < 2:
        st.error("Not enough assets downloaded.")
        return None

    prices = pd.DataFrame(px_list)
    prices = prices[prices.index <= pd.Timestamp(analysis_date)]
    prices = prices.ffill()

    # Resample to monthly (last trading day of each month)
    monthly = prices.resample("ME").last()

    # Drop assets without enough history
    min_rows = lookback_months + 4  # need extra for 1m/3m
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
            roll_max = cum.cummax()
            dd = ((cum - roll_max) / roll_max)
            max_dd = dd.min() * 100
            worst = r.min() * 100
            metrics[col] = {
                "Volatility %": round(vol, 2),
                "Max Drawdown %": round(max_dd, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Worst Month %": round(worst, 2)
            }
    return pd.DataFrame(metrics).T


def calc_forward_returns(symbol, analysis_date):
    """Calculate 1m and 3m forward returns from analysis date (for backtesting past dates)"""
    end_1m = analysis_date + timedelta(days=35)
    end_3m = analysis_date + timedelta(days=95)
    end = min(max(end_1m, end_3m), date.today())

    try:
        df = yf.download(symbol, start=analysis_date, end=end + timedelta(days=1),
                         auto_adjust=True, progress=False)
        if df.empty:
            return None, None

        prices = df["Close"].squeeze()
        base = prices.iloc[0]

        # 1 month forward (~21 trading days)
        ret_1m = None
        ret_3m = None

        if len(prices) >= 15:
            idx_1m = min(21, len(prices) - 1)
            ret_1m = round(((prices.iloc[idx_1m] / base) - 1) * 100, 2)

        if len(prices) >= 45:
            idx_3m = min(63, len(prices) - 1)
            ret_3m = round(((prices.iloc[idx_3m] / base) - 1) * 100, 2)

        return ret_1m, ret_3m
    except Exception:
        return None, None


# ---- Run Analysis ----
if run:
    CASH = "BIL"
    default_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD",
                       "SPY", "GLD", "IYR", "TLT", CASH]

    custom = []
    if custom_tickers_input.strip():
        custom = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()]

    all_symbols = list(dict.fromkeys(default_symbols + custom))
    is_past = analysis_date < date.today()

    monthly = get_monthly_prices(all_symbols, analysis_date, lookback_months)

    if monthly is not None:
        mom = calc_return(monthly, lookback_months)
        ret_1m = calc_return(monthly, 1)
        ret_3m = calc_return(monthly, 3)
        risk = calc_risk_metrics(monthly, lookback_months)

        # Build results table
        results = pd.DataFrame({
            "1M Return %": ret_1m,
            "3M Return %": ret_3m,
            f"{lookback_months}M Momentum %": mom,
        })
        results = results.join(risk)
        results = results.sort_values(f"{lookback_months}M Momentum %", ascending=False)

        # Strategy recommendation
        investable = [c for c in monthly.columns if c != CASH]
        mom_investable = mom[investable]
        best = mom_investable.idxmax()
        best_mom = mom_investable.max()

        if pd.notna(best_mom) and best_mom > 0:
            recommendation = best
            reason = f"highest positive momentum: +{best_mom:.2f}%"
        else:
            recommendation = CASH
            reason = "no assets show positive momentum — safety position"

        # ---- Recommendation Box ----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Analysis Date", str(analysis_date))
        with col2:
            st.metric("🏆 Recommended Asset", recommendation)
        with col3:
            st.metric("📊 Momentum", f"{best_mom:.2f}%" if pd.notna(best_mom) else "N/A")

        st.info(f"**Strategy says:** Invest in **{recommendation}** — {reason}")

        # ---- Forward Returns (past dates only) ----
        if is_past:
            st.markdown("---")
            st.subheader("⏩ Forward Returns from Analysis Date")
            st.caption("What actually happened after the strategy signal was given.")

            with st.spinner(f"Calculating forward returns for {recommendation}..."):
                fwd_1m, fwd_3m = calc_forward_returns(recommendation, analysis_date)

            fc1, fc2 = st.columns(2)
            with fc1:
                if fwd_1m is not None:
                    delta_color = "normal"
                    st.metric(
                        f"1 Month Return ({recommendation})",
                        f"{fwd_1m:+.2f}%",
                        delta=f"{'▲' if fwd_1m > 0 else '▼'} vs signal date"
                    )
                else:
                    st.metric("1 Month Return", "N/A", help="Not enough future data")

            with fc2:
                if fwd_3m is not None:
                    st.metric(
                        f"3 Month Return ({recommendation})",
                        f"{fwd_3m:+.2f}%",
                        delta=f"{'▲' if fwd_3m > 0 else '▼'} vs signal date"
                    )
                else:
                    st.metric("3 Month Return", "N/A", help="Not enough future data")

        st.markdown("---")

        # ---- Tabs ----
        tab1, tab2 = st.tabs(["📊 Momentum Strategy", "⚠️ Risk Analysis"])

        with tab1:
            st.subheader("Performance & Risk Matrix")

            def color_momentum(val):
                if pd.isna(val):
                    return ""
                return "color: green; font-weight: bold" if val > 0 else "color: red; font-weight: bold"

            def color_drawdown(val):
                if pd.isna(val):
                    return ""
                if val < -20:
                    return "background-color: #FF6B6B"
                elif val < -10:
                    return "background-color: #FFD700"
                return "background-color: #90EE90"

            styled = results.style\
                .applymap(color_momentum, subset=[f"{lookback_months}M Momentum %", "1M Return %", "3M Return %"])\
                .applymap(color_drawdown, subset=["Max Drawdown %"])\
                .format("{:.2f}", na_rep="N/A")

            st.dataframe(styled, use_container_width=True)

            st.subheader(f"Momentum Comparison ({lookback_months}M)")
            mom_df = results[[f"{lookback_months}M Momentum %"]].reset_index()
            mom_df.columns = ["Asset", "Momentum %"]
            mom_df["Color"] = mom_df["Momentum %"].apply(lambda x: "Positive" if x > 0 else "Negative")

            fig = px.bar(
                mom_df, x="Asset", y="Momentum %",
                color="Color",
                color_discrete_map={"Positive": "#90EE90", "Negative": "#FF6B6B"},
                title=f"{lookback_months}-Month Momentum by Asset"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Annualized Volatility")
                vol_df = results[["Volatility %"]].reset_index()
                vol_df.columns = ["Asset", "Volatility %"]
                fig2 = px.bar(vol_df, x="Asset", y="Volatility %",
                              color="Volatility %", color_continuous_scale="Reds",
                              title="Volatility (Higher = Riskier)")
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                st.subheader("Maximum Drawdown")
                dd_df = results[["Max Drawdown %"]].reset_index()
                dd_df.columns = ["Asset", "Max Drawdown %"]
                fig3 = px.bar(dd_df, x="Asset", y="Max Drawdown %",
                              color="Max Drawdown %", color_continuous_scale="Blues_r",
                              title="Max Drawdown (More Negative = Higher Risk)")
                st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Risk-Return Scatter")
            scatter_df = results[[f"{lookback_months}M Momentum %", "Volatility %"]].reset_index().dropna()
            scatter_df.columns = ["Asset", "Momentum %", "Volatility %"]
            fig4 = px.scatter(
                scatter_df, x="Volatility %", y="Momentum %",
                text="Asset", color="Asset",
                title="Risk vs Return — ideal assets are top-left"
            )
            fig4.update_traces(textposition="top center", marker_size=12)
            st.plotly_chart(fig4, use_container_width=True)

        st.caption(f"Data via Yahoo Finance. Analysis as of {analysis_date}. Not financial advice.")
