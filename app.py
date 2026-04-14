"""
Stock Comparison and Analysis Application
Financial Data Analytics Project
Run with: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import math
from scipy import stats

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="StockScope — Multi-Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

  .block-container { padding-top: 1.5rem; }

  div[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
  }
  div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #38bdf8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e3a5f;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
    color: #64748b;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
  }

  .sidebar-section {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
  }

  .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
TRADING_DAYS = 252
BENCHMARK = "^GSPC"
BENCHMARK_LABEL = "S&P 500"
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_COLORS = [
    "#38bdf8", "#f472b6", "#34d399", "#fb923c",
    "#a78bfa", "#facc15", "#f87171",
]

# ── Helpers ────────────────────────────────────────────────────────────────

def color_for(i: int) -> str:
    return ACCENT_COLORS[i % len(ACCENT_COLORS)]


def ann_return(daily_ret: pd.Series) -> float:
    """Annualised mean return (arithmetic)."""
    return float(daily_ret.mean() * TRADING_DAYS)


def ann_vol(daily_ret: pd.Series) -> float:
    """Annualised volatility."""
    return float(daily_ret.std() * math.sqrt(TRADING_DAYS))


# ── Data loading ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_prices(tickers: tuple, start: str, end: str) -> tuple[pd.DataFrame, list]:
    """
    Download adjusted closing prices for *tickers* + benchmark.
    Returns (prices_df, failed_tickers).
    prices_df columns = ticker symbols; index = dates.
    """
    all_tickers = list(tickers) + [BENCHMARK]
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        timeout=30,
    )

    # yfinance returns MultiIndex when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    # Identify failed tickers
    failed = [t for t in all_tickers if t not in prices.columns or prices[t].isna().all()]
    prices = prices.drop(columns=[t for t in failed if t in prices.columns])

    # Drop rows where ALL user tickers are NaN
    user_cols = [c for c in prices.columns if c != BENCHMARK and c in tickers]
    prices = prices.dropna(subset=user_cols, how="all")

    return prices, failed


def validate_and_trim(prices: pd.DataFrame, tickers: list, threshold: float = 0.05):
    """
    For each user ticker, check missing data fraction against threshold.
    Returns (trimmed_df, dropped_tickers, warnings).
    Uses overlapping date range approach.
    """
    warnings = []
    dropped = []

    # Overlap approach: find the date range where all user tickers have data
    user_cols = [t for t in tickers if t in prices.columns]
    overlap = prices[user_cols].dropna(how="any")
    if len(overlap) < len(prices) * 0.5:
        # Overlap is very short — try dropping tickers with >threshold missing
        for t in user_cols:
            miss_frac = prices[t].isna().mean()
            if miss_frac > threshold:
                dropped.append(t)
                warnings.append(
                    f"**{t}** dropped: {miss_frac:.0%} of dates missing (>{threshold:.0%} threshold)."
                )
        remaining = [t for t in user_cols if t not in dropped]
        overlap = prices[remaining + [BENCHMARK]].dropna(how="any") if remaining else pd.DataFrame()
    else:
        if len(overlap) < len(prices):
            warnings.append(
                f"Date range trimmed to overlapping period: "
                f"{overlap.index[0].date()} → {overlap.index[-1].date()}."
            )
        if BENCHMARK in prices.columns:
            overlap = overlap.join(prices[BENCHMARK].rename(BENCHMARK), how="left")

    return overlap, dropped, warnings


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 StockScope")
    st.markdown("---")

    # Ticker input
    raw_input = st.text_input(
        "Ticker symbols (2–5, comma-separated)",
        value="AAPL, MSFT, GOOGL",
        help="e.g. AAPL, MSFT, TSLA, AMZN",
    )

    # Date range
    default_start = date.today() - timedelta(days=5 * 365)
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("Start", value=default_start)
    end_date = col_e.date_input("End", value=date.today())

    run_btn = st.button("▶ Analyse", type="primary", use_container_width=True)

    st.markdown("---")

    # About / Methodology expander
    with st.expander("ℹ️ About & Methodology"):
        st.markdown("""
**What this app does**

StockScope lets you compare 2–5 stocks across four analysis pillars:
*Returns, Risk/Distribution, Correlation,* and *Portfolio Exploration*.

**Key assumptions**
- Returns are **simple arithmetic** daily returns: $r_t = (P_t / P_{t-1}) - 1$
- Annualisation uses **252 trading days**
- Prices are **adjusted closing prices** (splits & dividends adjusted)

**Data source**
Yahoo Finance via `yfinance`. The S&P 500 (`^GSPC`) is included as a benchmark for comparison only.

**Caching**
Data is cached for 1 hour to avoid redundant downloads.
        """)

# ── Input validation ───────────────────────────────────────────────────────

# Parse tickers
raw_tickers = [t.strip().upper() for t in raw_input.replace(",", " ").split() if t.strip()]
raw_tickers = list(dict.fromkeys(raw_tickers))  # deduplicate, preserve order

input_ok = True

if not run_btn and "prices_df" not in st.session_state:
    st.info("👈 Enter ticker symbols in the sidebar and click **▶ Analyse** to begin.")
    st.stop()

if run_btn:
    if len(raw_tickers) < 2:
        st.sidebar.error("Please enter at least **2** ticker symbols.")
        input_ok = False
    elif len(raw_tickers) > 5:
        st.sidebar.error("Please enter **no more than 5** ticker symbols.")
        input_ok = False

    if start_date >= end_date:
        st.sidebar.error("Start date must be **before** end date.")
        input_ok = False

    min_end = start_date + timedelta(days=365)
    if end_date < min_end:
        st.sidebar.error("Date range must span **at least 1 year**.")
        input_ok = False

    if input_ok:
        with st.spinner("Downloading data from Yahoo Finance…"):
            prices_raw, failed = load_prices(
                tuple(raw_tickers),
                str(start_date),
                str(end_date),
            )

        if failed:
            failed_user = [t for t in failed if t != BENCHMARK]
            if failed_user:
                st.error(
                    f"Could not download data for: **{', '.join(failed_user)}**. "
                    "Check the ticker symbol(s) and try again."
                )
                if BENCHMARK in failed:
                    st.warning("S&P 500 benchmark data also unavailable.")
                if not any(t in prices_raw.columns for t in raw_tickers):
                    st.stop()
            elif BENCHMARK in failed:
                st.warning("S&P 500 benchmark data unavailable; benchmark comparisons omitted.")

        # Validate & trim
        valid_tickers = [t for t in raw_tickers if t not in failed]
        prices_trimmed, dropped, val_warnings = validate_and_trim(prices_raw, valid_tickers)

        for w in val_warnings:
            st.warning(w)

        final_tickers = [t for t in valid_tickers if t not in dropped]

        if len(final_tickers) < 2:
            st.error("Fewer than 2 valid tickers remain after data validation. Please revise your inputs.")
            st.stop()

        if prices_trimmed.empty or len(prices_trimmed) < 10:
            st.error("Insufficient data returned. Try a wider date range or different tickers.")
            st.stop()

        # Store in session state so we don't re-download on widget interactions
        st.session_state["prices_df"] = prices_trimmed
        st.session_state["tickers"] = final_tickers

# ── Load from session state ────────────────────────────────────────────────

if "prices_df" not in st.session_state:
    st.stop()

prices: pd.DataFrame = st.session_state["prices_df"]
tickers: list = st.session_state["tickers"]

has_benchmark = BENCHMARK in prices.columns

# Compute daily returns for all columns
returns = prices.pct_change().dropna()
user_returns = returns[tickers]

# ── Header ─────────────────────────────────────────────────────────────────

st.markdown(f"### Analysing: `{'  ·  '.join(tickers)}`")
st.caption(
    f"Period: **{prices.index[0].date()}** → **{prices.index[-1].date()}**  "
    f"({len(prices):,} trading days)"
)

# ── Summary metrics row ────────────────────────────────────────────────────

cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    ret = ann_return(user_returns[t])
    vol = ann_vol(user_returns[t])
    total = float(prices[t].iloc[-1] / prices[t].iloc[0] - 1)
    cols[i].metric(
        label=t,
        value=f"{total:+.1%} total",
        delta=f"σ {vol:.1%} ann. vol",
    )

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Returns",
    "📉 Risk & Distribution",
    "🔗 Correlation",
    "🧮 Portfolio Explorer",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Return Analysis
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Price & Return Analysis")

    # Stock selector for price chart
    selected_for_price = st.multiselect(
        "Stocks to display on price chart",
        options=tickers,
        default=tickers,
        key="price_select",
    )

    if not selected_for_price:
        st.warning("Select at least one stock to display the price chart.")
    else:
        # ── Closing price chart ────────────────────────────────────────────
        fig_price = go.Figure()
        for i, t in enumerate(selected_for_price):
            fig_price.add_trace(go.Scatter(
                x=prices.index, y=prices[t],
                mode="lines", name=t,
                line=dict(color=color_for(tickers.index(t)), width=1.8),
            ))
        fig_price.update_layout(
            title="Adjusted Closing Prices",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template=PLOTLY_TEMPLATE,
            height=420,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # ── Summary statistics ─────────────────────────────────────────────────
    st.subheader("Summary Statistics")

    stat_cols = tickers + ([BENCHMARK_LABEL] if has_benchmark else [])
    stat_data = {}

    for t in tickers:
        r = user_returns[t].dropna()
        stat_data[t] = {
            "Ann. Return": f"{ann_return(r):.2%}",
            "Ann. Volatility": f"{ann_vol(r):.2%}",
            "Skewness": f"{float(r.skew()):.3f}",
            "Kurtosis": f"{float(r.kurt()):.3f}",
            "Min Daily Ret": f"{float(r.min()):.2%}",
            "Max Daily Ret": f"{float(r.max()):.2%}",
        }

    if has_benchmark:
        r = returns[BENCHMARK].dropna()
        stat_data[BENCHMARK_LABEL] = {
            "Ann. Return": f"{ann_return(r):.2%}",
            "Ann. Volatility": f"{ann_vol(r):.2%}",
            "Skewness": f"{float(r.skew()):.3f}",
            "Kurtosis": f"{float(r.kurt()):.3f}",
            "Min Daily Ret": f"{float(r.min()):.2%}",
            "Max Daily Ret": f"{float(r.max()):.2%}",
        }

    st.dataframe(
        pd.DataFrame(stat_data).T,
        use_container_width=True,
    )

    # ── Cumulative wealth index ────────────────────────────────────────────
    st.subheader("Cumulative Wealth Index — $10,000 Invested")

    wealth = (1 + user_returns).cumprod() * 10_000
    eq_weight = user_returns.mean(axis=1)
    wealth["Equal-Weight"] = (1 + eq_weight).cumprod() * 10_000
    if has_benchmark:
        bench_ret = returns[BENCHMARK].reindex(user_returns.index).fillna(0)
        wealth[BENCHMARK_LABEL] = (1 + bench_ret).cumprod() * 10_000

    fig_wealth = go.Figure()
    for i, col in enumerate(wealth.columns):
        dash = "dash" if col in (BENCHMARK_LABEL, "Equal-Weight") else "solid"
        width = 1.5 if col not in (BENCHMARK_LABEL, "Equal-Weight") else 1.2
        c = color_for(i) if col not in (BENCHMARK_LABEL, "Equal-Weight") else ("#94a3b8" if col == BENCHMARK_LABEL else "#facc15")
        fig_wealth.add_trace(go.Scatter(
            x=wealth.index, y=wealth[col],
            mode="lines", name=col,
            line=dict(color=c, width=width, dash=dash),
        ))
    fig_wealth.update_layout(
        title="Growth of $10,000",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template=PLOTLY_TEMPLATE,
        height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    # ── Box plot ───────────────────────────────────────────────────────────
    st.subheader("Daily Return Distribution — Boxplot")

    fig_box = go.Figure()
    for i, t in enumerate(tickers):
        fig_box.add_trace(go.Box(
            y=user_returns[t].dropna(),
            name=t,
            marker_color=color_for(i),
            boxmean="sd",
        ))
    fig_box.update_layout(
        title="Daily Returns by Stock",
        yaxis_title="Daily Return",
        template=PLOTLY_TEMPLATE,
        height=400,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk & Distribution Analysis
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Risk & Distribution Analysis")

    # ── Rolling volatility ─────────────────────────────────────────────────
    st.markdown("#### Rolling Annualised Volatility")
    roll_window = st.select_slider(
        "Rolling window (days)",
        options=[20, 30, 60, 90, 120],
        value=30,
        key="roll_vol_window",
    )

    fig_rvol = go.Figure()
    for i, t in enumerate(tickers):
        rvol = user_returns[t].rolling(roll_window).std() * math.sqrt(TRADING_DAYS)
        fig_rvol.add_trace(go.Scatter(
            x=rvol.index, y=rvol,
            mode="lines", name=t,
            line=dict(color=color_for(i), width=1.8),
        ))
    if has_benchmark:
        rvol_b = returns[BENCHMARK].rolling(roll_window).std() * math.sqrt(TRADING_DAYS)
        fig_rvol.add_trace(go.Scatter(
            x=rvol_b.index, y=rvol_b,
            mode="lines", name=BENCHMARK_LABEL,
            line=dict(color="#94a3b8", width=1.2, dash="dash"),
        ))
    fig_rvol.update_layout(
        title=f"{roll_window}-Day Rolling Annualised Volatility",
        xaxis_title="Date",
        yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        template=PLOTLY_TEMPLATE,
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_rvol, use_container_width=True)

    # ── Distribution plots ─────────────────────────────────────────────────
    st.markdown("#### Return Distribution")

    dist_stock = st.selectbox("Select stock for distribution analysis", tickers, key="dist_stock")
    dist_ret = user_returns[dist_stock].dropna()

    # Normality test
    jb_stat, jb_p = stats.jarque_bera(dist_ret)
    norm_verdict = (
        f"🔴 Rejects normality (p = {jb_p:.4f} < 0.05)"
        if jb_p < 0.05
        else f"🟢 Fails to reject normality (p = {jb_p:.4f} ≥ 0.05)"
    )

    col_jb1, col_jb2 = st.columns(2)
    col_jb1.metric("Jarque-Bera Statistic", f"{jb_stat:,.2f}")
    col_jb2.metric("p-value", f"{jb_p:.4f}")
    st.caption(f"**Normality test result:** {norm_verdict}")

    # Toggle histogram vs Q-Q
    dist_view = st.radio(
        "View",
        ["Histogram + Normal Fit", "Q-Q Plot"],
        horizontal=True,
        key="dist_view",
    )

    if dist_view == "Histogram + Normal Fit":
        mu, sigma = stats.norm.fit(dist_ret)
        x_range = np.linspace(dist_ret.min(), dist_ret.max(), 300)
        pdf_vals = stats.norm.pdf(x_range, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=dist_ret,
            histnorm="probability density",
            name="Observed",
            marker_color=color_for(tickers.index(dist_stock)),
            opacity=0.65,
            nbinsx=60,
        ))
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=pdf_vals,
            mode="lines", name="Normal Fit",
            line=dict(color="#f8fafc", width=2),
        ))
        fig_hist.update_layout(
            title=f"{dist_stock} — Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            xaxis_tickformat=".1%",
            template=PLOTLY_TEMPLATE,
            height=420,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:  # Q-Q plot
        qq = stats.probplot(dist_ret, dist="norm")
        theoretical_q, sample_q = qq[0]
        slope, intercept, _ = qq[1]
        line_x = np.array([theoretical_q.min(), theoretical_q.max()])
        line_y = slope * line_x + intercept

        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=sample_q,
            mode="markers", name="Quantiles",
            marker=dict(color=color_for(tickers.index(dist_stock)), size=4, opacity=0.7),
        ))
        fig_qq.add_trace(go.Scatter(
            x=line_x, y=line_y,
            mode="lines", name="Normal Reference",
            line=dict(color="#f8fafc", width=1.5, dash="dash"),
        ))
        fig_qq.update_layout(
            title=f"{dist_stock} — Q-Q Plot vs. Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template=PLOTLY_TEMPLATE,
            height=420,
        )
        st.plotly_chart(fig_qq, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Correlation & Diversification
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Correlation & Diversification")

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.markdown("#### Pairwise Correlation Heatmap")

    corr_matrix = user_returns.corr()

    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
    ))
    fig_heat.update_layout(
        title="Correlation Matrix of Daily Returns",
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Scatter plot ───────────────────────────────────────────────────────
    st.markdown("#### Return Scatter Plot")

    s_col1, s_col2 = st.columns(2)
    scatter_a = s_col1.selectbox("Stock A", tickers, index=0, key="scatter_a")
    scatter_b = s_col2.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="scatter_b")

    if scatter_a == scatter_b:
        st.warning("Select two different stocks for the scatter plot.")
    else:
        corr_val = float(user_returns[scatter_a].corr(user_returns[scatter_b]))
        st.caption(f"Pearson correlation: **{corr_val:.4f}**")

        fig_scat = px.scatter(
            x=user_returns[scatter_a],
            y=user_returns[scatter_b],
            labels={"x": f"{scatter_a} Daily Return", "y": f"{scatter_b} Daily Return"},
            title=f"{scatter_a} vs {scatter_b} — Daily Returns",
            trendline="ols",
            template=PLOTLY_TEMPLATE,
        )
        fig_scat.update_traces(
            marker=dict(size=4, opacity=0.5, color=color_for(0)),
            selector=dict(mode="markers"),
        )
        fig_scat.update_layout(height=420)
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Rolling correlation ────────────────────────────────────────────────
    st.markdown("#### Rolling Correlation")

    rc_col1, rc_col2, rc_col3 = st.columns(3)
    rc_a = rc_col1.selectbox("Stock A", tickers, index=0, key="rc_a")
    rc_b = rc_col2.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="rc_b")
    rc_window = rc_col3.select_slider("Window (days)", options=[20, 30, 60, 90, 120], value=60, key="rc_window")

    if rc_a == rc_b:
        st.warning("Select two different stocks for rolling correlation.")
    else:
        roll_corr = user_returns[rc_a].rolling(rc_window).corr(user_returns[rc_b])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            mode="lines", name=f"{rc_a} / {rc_b}",
            line=dict(color=color_for(2), width=1.8),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.1)",
        ))
        fig_rc.add_hline(y=0, line=dict(color="#64748b", dash="dot", width=1))
        fig_rc.update_layout(
            title=f"{rc_window}-Day Rolling Correlation: {rc_a} vs {rc_b}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis=dict(range=[-1.1, 1.1]),
            template=PLOTLY_TEMPLATE,
            height=380,
        )
        st.plotly_chart(fig_rc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Two-Asset Portfolio Explorer
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Two-Asset Portfolio Explorer")

    st.info(
        "**Diversification in action.** Combining two assets can produce a portfolio with "
        "*lower* volatility than either stock individually — especially when their returns "
        "are less than perfectly correlated (ρ < 1). Adjust the weight slider to explore "
        "the volatility curve across all possible allocations.",
        icon="ℹ️",
    )

    p_col1, p_col2 = st.columns(2)
    port_a = p_col1.selectbox("Stock A", tickers, index=0, key="port_a")
    port_b = p_col2.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="port_b")

    if port_a == port_b:
        st.warning("Select two different stocks for portfolio analysis.")
    else:
        weight_a = st.slider(
            f"Weight on {port_a} (remainder goes to {port_b})",
            min_value=0, max_value=100, value=50, step=1,
            format="%d%%",
            key="port_weight",
        )
        weight_b = 100 - weight_a
        wa = weight_a / 100
        wb = weight_b / 100

        ra = user_returns[port_a].dropna()
        rb = user_returns[port_b].dropna()
        ra, rb = ra.align(rb, join="inner")

        mu_a = ann_return(ra)
        mu_b = ann_return(rb)
        sig_a = ann_vol(ra)
        sig_b = ann_vol(rb)
        rho = float(ra.corr(rb))
        cov_ab = rho * sig_a * sig_b

        # Current portfolio stats
        port_ret = wa * mu_a + wb * mu_b
        port_vol = math.sqrt(wa**2 * sig_a**2 + wb**2 * sig_b**2 + 2 * wa * wb * cov_ab)

        # Display metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{port_a} Weight", f"{weight_a}%")
        m2.metric(f"{port_b} Weight", f"{weight_b}%")
        m3.metric("Portfolio Ann. Return", f"{port_ret:.2%}")
        m4.metric("Portfolio Ann. Volatility", f"{port_vol:.2%}")

        # Full volatility curve
        weights_range = np.linspace(0, 1, 201)
        vols_curve = np.sqrt(
            weights_range**2 * sig_a**2
            + (1 - weights_range)**2 * sig_b**2
            + 2 * weights_range * (1 - weights_range) * cov_ab
        )
        rets_curve = weights_range * mu_a + (1 - weights_range) * mu_b

        fig_port = go.Figure()

        # Full curve
        fig_port.add_trace(go.Scatter(
            x=weights_range * 100, y=vols_curve,
            mode="lines", name="Portfolio Volatility",
            line=dict(color="#38bdf8", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.07)",
        ))

        # Individual stock vol lines
        fig_port.add_hline(
            y=sig_a, line=dict(color=color_for(0), dash="dash", width=1.2),
            annotation_text=f"{port_a} σ = {sig_a:.1%}",
            annotation_position="right",
        )
        fig_port.add_hline(
            y=sig_b, line=dict(color=color_for(1), dash="dash", width=1.2),
            annotation_text=f"{port_b} σ = {sig_b:.1%}",
            annotation_position="right",
        )

        # Minimum variance point
        min_vol_idx = int(np.argmin(vols_curve))
        fig_port.add_trace(go.Scatter(
            x=[weights_range[min_vol_idx] * 100],
            y=[vols_curve[min_vol_idx]],
            mode="markers", name="Min Variance",
            marker=dict(color="#34d399", size=12, symbol="diamond"),
        ))

        # Current slider position
        fig_port.add_trace(go.Scatter(
            x=[weight_a],
            y=[port_vol],
            mode="markers+text", name="Current Allocation",
            marker=dict(color="#fb923c", size=14, symbol="circle"),
            text=[f"  {weight_a}% / {weight_b}%"],
            textposition="middle right",
            textfont=dict(color="#fb923c"),
        ))

        fig_port.update_layout(
            title=f"Portfolio Volatility Curve: {port_a} ({weight_a}%) + {port_b} ({weight_b}%)",
            xaxis_title=f"Weight on {port_a} (%)",
            yaxis_title="Annualised Volatility",
            yaxis_tickformat=".0%",
            template=PLOTLY_TEMPLATE,
            height=460,
            legend=dict(orientation="h", y=-0.15),
            annotations=[
                dict(
                    x=weights_range[min_vol_idx] * 100,
                    y=vols_curve[min_vol_idx],
                    text=f"Min σ = {vols_curve[min_vol_idx]:.1%}<br>({weights_range[min_vol_idx]*100:.0f}% {port_a})",
                    showarrow=True, arrowhead=2,
                    ax=30, ay=-40,
                    font=dict(color="#34d399"),
                    arrowcolor="#34d399",
                )
            ],
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # Supplementary table
        st.markdown("##### Individual Stock Statistics")
        ind_df = pd.DataFrame({
            "Ticker": [port_a, port_b],
            "Ann. Return": [f"{mu_a:.2%}", f"{mu_b:.2%}"],
            "Ann. Volatility": [f"{sig_a:.2%}", f"{sig_b:.2%}"],
            "Correlation (ρ)": [f"{rho:.4f}", f"{rho:.4f}"],
        }).set_index("Ticker")
        st.dataframe(ind_df, use_container_width=True)

        st.markdown(
            f"> **Correlation between {port_a} and {port_b}: ρ = {rho:.4f}.** "
            f"{'Since ρ < 1, combining these stocks reduces portfolio volatility below a simple weighted average of individual volatilities — the diversification benefit is real.' if rho < 0.99 else 'These assets are nearly perfectly correlated, so diversification benefits are minimal.'}"
        )