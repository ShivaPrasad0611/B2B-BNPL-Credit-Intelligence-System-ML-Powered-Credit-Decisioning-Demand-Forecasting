"""
B2B BNPL — Credit Predictor + Demand Forecasting
Run: streamlit run b2b_bnpl_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="B2B BNPL Intelligence",
    page_icon="💳",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0f1117; color: #e0e0e0; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

.header { text-align:center; padding:28px 0 6px; }
.header h1 { color:#4fc3f7; font-size:2rem; margin-bottom:4px; }
.header p  { color:#90a4ae; font-size:0.9rem; }

.sec {
    color:#4fc3f7; font-weight:700; font-size:0.82rem;
    text-transform:uppercase; letter-spacing:0.06em;
    border-bottom:1px solid #2d3561;
    padding-bottom:4px; margin:22px 0 12px;
}

.card {
    border-radius:14px; padding:24px 20px;
    text-align:center; margin:16px 0 10px; border:2px solid;
}
.approved { background:#0a2b0a; border-color:#43a047; }
.rejected { background:#2b0a0a; border-color:#e53935; }
.card h2 { font-size:2rem; margin:0 0 4px; }
.card p  { color:#90a4ae; margin:0; font-size:0.85rem; }

.kpis { display:flex; gap:12px; margin:14px 0; }
.kpi  {
    flex:1; background:#1a1f2e; border:1px solid #2d3561;
    border-radius:10px; padding:14px; text-align:center;
}
.kpi h3 { color:#4fc3f7; font-size:1.45rem; margin:0; }
.kpi p  { color:#90a4ae; font-size:0.72rem; margin:4px 0 0; }

.forecast-card {
    background:#111827; border:1px solid #2d3561;
    border-radius:12px; padding:20px; margin:8px 0;
    text-align:center;
}
.forecast-card h2 { color:#4fc3f7; font-size:1.8rem; margin:0 4px; }
.forecast-card p  { color:#90a4ae; font-size:0.8rem; margin:4px 0 0; }

.trend-up   { color:#43a047 !important; }
.trend-down { color:#e53935 !important; }

.stButton > button {
    width:100%;
    background:linear-gradient(90deg,#1565c0,#0d47a1);
    color:white; border:none; border-radius:10px;
    padding:13px; font-size:1rem; font-weight:700;
}
.stButton > button:hover {
    background:linear-gradient(90deg,#1976d2,#1565c0);
}
</style>
""", unsafe_allow_html=True)

BG   = "#0f1117"
CARD = "#1a1f2e"

# ── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_models():
    clf          = joblib.load("bnpl_classifier.pkl")
    risk_reg     = joblib.load("bnpl_risk_regressor.pkl")
    credit_feats = joblib.load("bnpl_credit_features.pkl")
    order_reg    = joblib.load("bnpl_order_forecast.pkl")
    gmv_reg      = joblib.load("bnpl_gmv_forecast.pkl")
    demand_feats = joblib.load("bnpl_demand_features.pkl")
    return clf, risk_reg, credit_feats, order_reg, gmv_reg, demand_feats

try:
    clf, risk_reg, credit_feats, order_reg, gmv_reg, demand_feats = load_models()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run **train_model.ipynb** first.")
    st.stop()

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>💳 B2B BNPL Intelligence Platform</h1>
    <p>Credit Prediction &nbsp;|&nbsp; Demand Forecasting</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💳  Credit Predictor", "📦  Demand Forecasting"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — CREDIT PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Enter your business details to get an instant BNPL credit decision")
    st.divider()

    st.markdown('<div class="sec">🏢 About Your Business</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        biz_age   = st.number_input("Business Age (years)",    0.5, 50.0,  5.0, 0.5, key="c_age")
        employees = st.number_input("Number of Employees",     1,   1000,  15,       key="c_emp")
    with c2:
        revenue   = st.number_input("Annual Revenue (₹ Lakh)", 1.0, 10000.0, 200.0,  key="c_rev")
        gst       = st.radio("GST Registered?", ["Yes", "No"], horizontal=True, key="c_gst")

    st.markdown('<div class="sec">💰 Financial Health</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        gross_margin  = st.slider("Gross Margin %",       0.0, 80.0, 20.0, key="c_gm",
                                  help="(Revenue − Cost of Goods) / Revenue × 100")
        current_ratio = st.slider("Current Ratio",        0.3,  8.0,  1.5, key="c_cr",
                                  help="Current Assets ÷ Current Liabilities. Above 1 = healthy")
    with c4:
        debt_eq       = st.slider("Debt-to-Equity Ratio", 0.0,  8.0,  1.0, key="c_de",
                                  help="Total Debt ÷ Equity. Lower is better")
        existing_loan = st.number_input("Existing Loan Amount (₹ Lakh)", 0.0, 1000.0, 0.0, key="c_el")

    st.markdown('<div class="sec">🧾 Payment Track Record</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        pct_ontime = st.slider("Invoices Paid On Time %",      0.0, 100.0, 80.0, key="c_ot")
        avg_delay  = st.slider("Average Payment Delay (days)",  0.0,  60.0,  5.0, key="c_ad")
    with c6:
        months_plat = st.slider("Months Active on Platform",   1, 100, 12, key="c_mp")

    st.markdown('<div class="sec">📋 Credit Background</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        bureau_score  = st.slider("Credit Bureau Score (CIBIL)", 300, 900, 700, key="c_bs")
        prev_defaults = st.number_input("Past BNPL Defaults", 0, 10, 0, key="c_pd")
    with c8:
        enquiries = st.slider("Loan Enquiries (Last 6 Months)", 0, 10, 1, key="c_enq")

    st.markdown("")
    credit_go = st.button("⚡  Get Credit Decision", key="btn_credit")

    if credit_go:
        row = {
            "business_age_years":        biz_age,
            "annual_revenue_lakh":       revenue,
            "num_employees":             employees,
            "gross_margin_pct":          gross_margin,
            "debt_to_equity_ratio":      debt_eq,
            "current_ratio":             current_ratio,
            "gst_registered":            1 if gst == "Yes" else 0,
            "pct_invoices_paid_on_time": pct_ontime,
            "avg_payment_delay_days":    avg_delay,
            "months_on_platform":        months_plat,
            "credit_bureau_score":       bureau_score,
            "prev_bnpl_defaults":        prev_defaults,
            "num_credit_enquiries_6m":   enquiries,
            "existing_loan_lakh":        existing_loan,
        }
        Xi       = pd.DataFrame([row])[credit_feats]
        approved = int(clf.predict(Xi)[0])
        prob     = float(clf.predict_proba(Xi)[0, 1])
        def_risk = float(risk_reg.predict(Xi)[0])

        limit = 0.0
        if approved:
            limit = max(round(
                revenue * 0.15
                * ((bureau_score - 300) / 600)
                * min(months_plat / 24, 1.0)
                * (1 - def_risk), 2), 2.0)

        st.divider()
        st.markdown("## 🎯 Credit Decision")

        if approved:
            st.markdown("""
            <div class="card approved">
                <h2 style="color:#43a047">✅ APPROVED</h2>
                <p>Your business qualifies for B2B BNPL credit</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card rejected">
                <h2 style="color:#e53935">❌ NOT APPROVED</h2>
                <p>Your profile does not meet current BNPL criteria</p>
            </div>""", unsafe_allow_html=True)

        rc = "#e53935" if def_risk > 0.5 else ("#fb8c00" if def_risk > 0.3 else "#43a047")
        st.markdown(f"""
        <div class="kpis">
            <div class="kpi"><h3>{prob*100:.1f}%</h3><p>Approval Probability</p></div>
            <div class="kpi"><h3 style="color:{rc}">{def_risk*100:.1f}%</h3><p>Default Risk Score</p></div>
            <div class="kpi"><h3>₹{limit:.1f}L</h3><p>Recommended Credit Limit</p></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Approval Confidence**")
        st.progress(prob)

        st.markdown("### 📌 Factor Breakdown")
        checks = [
            ("Credit Bureau Score",   f"{bureau_score}",
             bureau_score >= 700,   600 <= bureau_score < 700),
            ("Invoices Paid On Time", f"{pct_ontime:.0f}%",
             pct_ontime >= 80,      70 <= pct_ontime < 80),
            ("Past BNPL Defaults",    str(int(prev_defaults)),
             prev_defaults == 0,    prev_defaults == 1),
            ("Debt-to-Equity Ratio",  f"{debt_eq:.2f}",
             debt_eq <= 1.5,        1.5 < debt_eq <= 3.0),
            ("Current Ratio",         f"{current_ratio:.2f}",
             current_ratio >= 1.2,  1.0 <= current_ratio < 1.2),
            ("Gross Margin",          f"{gross_margin:.1f}%",
             gross_margin >= 20,    10 <= gross_margin < 20),
        ]
        left, right = st.columns(2)
        for i, (lbl, val, good, warn) in enumerate(checks):
            icon  = "✅" if good else ("⚠️" if warn else "❌")
            color = "#43a047" if good else ("#fb8c00" if warn else "#e53935")
            (left if i % 2 == 0 else right).markdown(
                f"{icon} **{lbl}** — <span style='color:{color};font-weight:600'>{val}</span>",
                unsafe_allow_html=True)

        st.markdown("### 💡 How to Improve")
        tips = []
        if bureau_score  < 700: tips.append("📈 **Credit Score** — Pay off outstanding dues to push score above 700.")
        if pct_ontime    < 80:  tips.append("🧾 **On-Time Payments** — Target 80%+ on-time invoices.")
        if prev_defaults > 0:   tips.append("🚫 **Past Defaults** — Settle previous BNPL defaults before reapplying.")
        if debt_eq       > 1.5: tips.append("⚖️ **Debt Ratio** — Reduce total debt below 1.5× equity.")
        if months_plat   < 6:   tips.append("⏳ **Platform History** — Stay active for 6+ months first.")
        if existing_loan > 100: tips.append("💳 **Existing Loans** — High loan burden lowers eligibility.")
        if not tips:            tips.append("🌟 **Strong profile!** Keep paying on time to unlock higher limits.")
        for t in tips:
            st.info(t)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Enter your current business metrics to forecast next month's demand")
    st.divider()

    st.markdown('<div class="sec">📦 Current Order Activity</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        inv_freq    = st.number_input("Current Orders Per Month",        0.5, 200.0,  8.0, 0.5, key="d_freq")
        avg_inv_val = st.number_input("Average Order Value (₹ Lakh)",   0.1, 200.0,  3.0, 0.1, key="d_aiv")
    with d2:
        platform_gmv = st.number_input("Total Platform GMV So Far (₹ Lakh)", 0.0, 20000.0, 100.0, key="d_gmv")
        last_order   = st.slider("Days Since Last Order", 0, 180, 7, key="d_lo")

    st.markdown('<div class="sec">📈 Growth & Trends</div>', unsafe_allow_html=True)
    d3, d4 = st.columns(2)
    with d3:
        rev_growth = st.slider("Revenue Growth YoY %",      -50.0, 100.0, 15.0, key="d_rg")
        rev_vol    = st.slider("Revenue Volatility (6M) %",   0.0,  50.0, 10.0, key="d_rv")
    with d4:
        repeat_rate = st.slider("Repeat Order Rate %",   0.0, 100.0, 70.0, key="d_rr")
        order_trend = st.select_slider("Order Value Trend",
                                       options=["declining", "stable", "growing"],
                                       value="stable", key="d_ot")

    st.markdown('<div class="sec">🏢 Business Context</div>', unsafe_allow_html=True)
    d5, d6 = st.columns(2)
    with d5:
        months_plat2  = st.slider("Months Active on Platform", 1, 100, 12, key="d_mp")
        num_billers   = st.slider("Number of Unique Suppliers", 1,  20,  3, key="d_nb")
    with d6:
        annual_rev2   = st.number_input("Annual Revenue (₹ Lakh)", 1.0, 10000.0, 200.0, key="d_ar")
        gross_margin2 = st.slider("Gross Margin %",               0.0,  80.0,   20.0, key="d_gm2")

    st.markdown('<div class="sec">🌐 Market Context</div>', unsafe_allow_html=True)
    d7, d8 = st.columns(2)
    with d7:
        sector_bnpl = st.slider("Sector BNPL Adoption %", 0.0, 50.0, 20.0, key="d_sb")
    with d8:
        gdp_growth  = st.slider("Current GDP Growth %",   0.0, 15.0,  6.5, key="d_gdp")

    st.markdown("")
    demand_go = st.button("📦  Forecast Next Month Demand", key="btn_demand")

    if demand_go:
        trend_enc   = {"growing": 1, "stable": 0, "declining": -1}[order_trend]
        gmv_per_inv = platform_gmv / max(inv_freq * months_plat2, 1)

        d_row = {
            "months_on_platform":          months_plat2,
            "invoice_frequency_monthly":   inv_freq,
            "avg_invoice_value_lakh":      avg_inv_val,
            "repeat_order_rate_pct":       repeat_rate,
            "revenue_growth_yoy_pct":      rev_growth,
            "revenue_volatility_6m_pct":   rev_vol,
            "platform_gmv_lakh":           platform_gmv,
            "num_unique_billers":          num_billers,
            "last_order_days_ago":         last_order,
            "gross_margin_pct":            gross_margin2,
            "annual_revenue_lakh":         annual_rev2,
            "trend_enc":                   trend_enc,
            "gmv_per_invoice":             gmv_per_inv,
            "sector_bnpl_penetration_pct": sector_bnpl,
            "macro_gdp_growth_pct":        gdp_growth,
        }

        Xd          = pd.DataFrame([d_row])[demand_feats]
        next_orders = float(order_reg.predict(Xd)[0])
        next_gmv    = float(gmv_reg.predict(Xd)[0])

        current_gmv  = inv_freq * avg_inv_val
        gmv_change   = ((next_gmv    - current_gmv) / max(current_gmv, 0.01)) * 100
        order_change = ((next_orders - inv_freq)    / max(inv_freq,    0.01)) * 100

        # 6-month projection with slight tapering
        taper       = 0.6
        months      = ["Current", "M+1", "M+2", "M+3", "M+4", "M+5", "M+6"]
        gmv_proj    = [current_gmv]
        order_proj  = [inv_freq]
        g_rate      = gmv_change / 100
        o_rate      = order_change / 100
        for i in range(6):
            factor   = taper ** i
            gmv_proj.append(   round(gmv_proj[-1]   * (1 + g_rate * factor), 2))
            order_proj.append( round(order_proj[-1] * (1 + o_rate * factor), 2))

        # ── RESULT HEADER ──
        st.divider()
        st.markdown("## 📦 Demand Forecast Results")

        g_arrow = "↑" if gmv_change >= 0 else "↓"
        o_arrow = "↑" if order_change >= 0 else "↓"
        g_cls   = "trend-up" if gmv_change >= 0 else "trend-down"
        o_cls   = "trend-up" if order_change >= 0 else "trend-down"

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"""
            <div class="forecast-card">
                <p style="color:#90a4ae;font-size:0.8rem;margin:0">NEXT MONTH ORDERS</p>
                <h2>{next_orders:.1f}</h2>
                <p class="{o_cls}">{o_arrow} {abs(order_change):.1f}% vs this month</p>
            </div>""", unsafe_allow_html=True)
        with fc2:
            st.markdown(f"""
            <div class="forecast-card">
                <p style="color:#90a4ae;font-size:0.8rem;margin:0">NEXT MONTH GMV</p>
                <h2>₹{next_gmv:.1f}L</h2>
                <p class="{g_cls}">{g_arrow} {abs(gmv_change):.1f}% vs this month</p>
            </div>""", unsafe_allow_html=True)

        # ── CHART 1: GMV Forecast Line Chart ──────────────────
        st.markdown("### 📈 6-Month GMV Forecast")
        fig_gmv = go.Figure()

        # Shaded confidence band (±15%)
        upper = [v * 1.15 for v in gmv_proj]
        lower = [max(v * 0.85, 0) for v in gmv_proj]
        fig_gmv.add_trace(go.Scatter(
            x=months + months[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(79,195,247,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
            showlegend=False,
        ))
        # Main forecast line
        fig_gmv.add_trace(go.Scatter(
            x=months, y=gmv_proj,
            mode="lines+markers+text",
            line=dict(color="#4fc3f7", width=3),
            marker=dict(size=8, color="#4fc3f7",
                        line=dict(color="white", width=1.5)),
            text=[f"₹{v:.1f}L" for v in gmv_proj],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            name="Forecasted GMV",
        ))
        # Current month marker
        fig_gmv.add_trace(go.Scatter(
            x=["Current"], y=[current_gmv],
            mode="markers",
            marker=dict(size=12, color="#fb8c00",
                        symbol="diamond",
                        line=dict(color="white", width=2)),
            name="Current Month",
        ))
        fig_gmv.update_layout(
            plot_bgcolor=BG, paper_bgcolor=CARD,
            font_color="white", height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(gridcolor="#2d3561"),
            yaxis=dict(gridcolor="#2d3561", title="GMV (₹ Lakh)"),
        )
        st.plotly_chart(fig_gmv, use_container_width=True)

        # ── CHART 2: Order Count Bar Chart ────────────────────
        st.markdown("### 📊 6-Month Order Count Forecast")
        bar_colors = ["#fb8c00"] + [
            "#43a047" if v >= order_proj[0] else "#e53935"
            for v in order_proj[1:]
        ]
        fig_orders = go.Figure()
        fig_orders.add_trace(go.Bar(
            x=months, y=order_proj,
            marker_color=bar_colors,
            text=[f"{v:.1f}" for v in order_proj],
            textposition="outside",
            textfont=dict(color="white", size=11),
            name="Orders",
        ))
        # Target line = 10% above current
        target = inv_freq * 1.10
        fig_orders.add_hline(
            y=target, line_dash="dash",
            line_color="#4fc3f7", line_width=1.5,
            annotation_text=f"  +10% Target ({target:.1f})",
            annotation_font_color="#4fc3f7",
        )
        fig_orders.update_layout(
            plot_bgcolor=BG, paper_bgcolor=CARD,
            font_color="white", height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(gridcolor="#2d3561"),
            yaxis=dict(gridcolor="#2d3561", title="Orders / Month"),
            showlegend=False,
        )
        st.plotly_chart(fig_orders, use_container_width=True)

        # ── CHART 3: Revenue Mix Donut ─────────────────────────
        st.markdown("### 🥧 Revenue Composition")
        cogs          = annual_rev2 * (1 - gross_margin2 / 100) / 12
        gross_profit  = annual_rev2 * (gross_margin2 / 100) / 12
        bnpl_contrib  = next_gmv
        other_rev     = max((annual_rev2 / 12) - bnpl_contrib - gross_profit, 0)

        fig_donut = go.Figure(go.Pie(
            labels=["BNPL Platform GMV", "Gross Profit (est.)", "COGS (est.)", "Other Revenue"],
            values=[bnpl_contrib, gross_profit, cogs, other_rev],
            hole=0.55,
            marker_colors=["#4fc3f7", "#43a047", "#e53935", "#fb8c00"],
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_donut.update_layout(
            paper_bgcolor=CARD, font_color="white",
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # ── CHART 4: KPI Radar ────────────────────────────────
        st.markdown("### 🕸️ Business Health Radar")
        radar_labels = [
            "Repeat Rate", "Revenue Growth",
            "Margin Health", "Platform Maturity",
            "Supplier Diversity", "Order Stability"
        ]
        # Normalize each dimension 0-100
        radar_vals = [
            min(repeat_rate, 100),
            min(max((rev_growth + 50) / 1.5, 0), 100),
            min(gross_margin2 * 1.25, 100),
            min(months_plat2 * 2, 100),
            min(num_billers * 10, 100),
            max(100 - rev_vol * 2, 0),
        ]
        radar_labels_closed = radar_labels + [radar_labels[0]]
        radar_vals_closed   = radar_vals   + [radar_vals[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_closed,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(79,195,247,0.18)",
            line=dict(color="#4fc3f7", width=2),
            marker=dict(size=6, color="#4fc3f7"),
            name="Your Business",
        ))
        # Benchmark ring at 70
        bench = [70] * len(radar_labels_closed)
        fig_radar.add_trace(go.Scatterpolar(
            r=bench,
            theta=radar_labels_closed,
            fill="toself",
            fillcolor="rgba(67,160,71,0.06)",
            line=dict(color="#43a047", width=1.5, dash="dot"),
            name="Benchmark (70)",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#111827",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#2d3561", tickfont=dict(size=9)),
                angularaxis=dict(gridcolor="#2d3561"),
            ),
            paper_bgcolor=CARD, font_color="white",
            height=360, margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(orientation="h", y=-0.1),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── CHART 5: GMV vs Orders Dual Axis ──────────────────
        st.markdown("### 📉 GMV vs Orders — Trend Comparison")
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Scatter(x=months, y=gmv_proj,
                       mode="lines+markers",
                       line=dict(color="#4fc3f7", width=2.5),
                       marker=dict(size=7),
                       name="GMV (₹ Lakh)"),
            secondary_y=False,
        )
        fig_dual.add_trace(
            go.Scatter(x=months, y=order_proj,
                       mode="lines+markers",
                       line=dict(color="#fb8c00", width=2.5, dash="dot"),
                       marker=dict(size=7, symbol="square"),
                       name="Orders / Month"),
            secondary_y=True,
        )
        fig_dual.update_layout(
            plot_bgcolor=BG, paper_bgcolor=CARD,
            font_color="white", height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=-0.18),
            xaxis=dict(gridcolor="#2d3561"),
        )
        fig_dual.update_yaxes(title_text="GMV (₹ Lakh)",   gridcolor="#2d3561", secondary_y=False)
        fig_dual.update_yaxes(title_text="Orders / Month", gridcolor="#2d3561", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)

        # ── 6-MONTH TABLE ─────────────────────────────────────
        st.markdown("### 📅 6-Month Projection Table")
        proj_df = pd.DataFrame({
            "Month":         months,
            "Orders":        [round(v, 1) for v in order_proj],
            "GMV (₹ Lakh)":  [round(v, 2) for v in gmv_proj],
            "MoM GMV Δ":     ["—"] + [
                f"+{((gmv_proj[i]-gmv_proj[i-1])/max(gmv_proj[i-1],0.01)*100):.1f}%"
                if gmv_proj[i] >= gmv_proj[i-1]
                else f"{((gmv_proj[i]-gmv_proj[i-1])/max(gmv_proj[i-1],0.01)*100):.1f}%"
                for i in range(1, len(gmv_proj))
            ],
        })
        st.dataframe(proj_df, use_container_width=True, hide_index=True)

        # ── DEMAND SIGNALS ────────────────────────────────────
        st.markdown("### 📊 Demand Signals")
        signals = [
            ("Order Trend",       order_trend.capitalize(),
             order_trend == "growing", order_trend == "stable"),
            ("Repeat Rate",       f"{repeat_rate:.0f}%",
             repeat_rate >= 70,   50 <= repeat_rate < 70),
            ("Revenue Growth",    f"{rev_growth:.1f}%",
             rev_growth >= 10,    0 <= rev_growth < 10),
            ("Revenue Stability", f"{rev_vol:.1f}% volatility",
             rev_vol <= 10,       10 < rev_vol <= 20),
            ("Recency",           f"{last_order}d ago",
             last_order <= 7,     7 < last_order <= 30),
            ("Supplier Base",     f"{num_billers} suppliers",
             num_billers >= 3,    num_billers == 2),
        ]
        sl, sr = st.columns(2)
        for i, (lbl, val, good, warn) in enumerate(signals):
            icon  = "✅" if good else ("⚠️" if warn else "❌")
            color = "#43a047" if good else ("#fb8c00" if warn else "#e53935")
            (sl if i % 2 == 0 else sr).markdown(
                f"{icon} **{lbl}** — <span style='color:{color};font-weight:600'>{val}</span>",
                unsafe_allow_html=True)

        # ── TIPS ──────────────────────────────────────────────
        st.markdown("### 💡 Demand Improvement Tips")
        tips = []
        if repeat_rate  < 70:            tips.append("🔄 **Loyalty** — Strengthen supplier relationships to push repeat rate above 70%.")
        if rev_growth   < 5:             tips.append("📈 **Growth** — Explore new categories or geographies to drive revenue.")
        if rev_vol      > 20:            tips.append("📉 **Stability** — Secure recurring order contracts to reduce volatility.")
        if last_order   > 30:            tips.append("⚡ **Recency** — Re-engage with suppliers soon to stay active.")
        if num_billers  < 3:             tips.append("🌐 **Diversify** — Add more suppliers to reduce dependency risk.")
        if order_trend == "declining":   tips.append("⚠️ **Declining Trend** — Review pricing or target new buyer segments.")
        if not tips:                     tips.append("🌟 **Healthy demand profile!** Maintain consistency to unlock higher credit limits.")
        for t in tips:
            st.info(t)

# ── FOOTER ────────────────────────────────────────────────
st.divider()
st.caption("B2B BNPL Intelligence Platform · Credit Predictor + Demand Forecasting · Powered by XGBoost & Plotly")