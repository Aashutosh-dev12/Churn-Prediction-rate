import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight · Prediction",
    page_icon="⚡",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
  --bg:        #050810;
  --surface:   rgba(255,255,255,0.04);
  --border:    rgba(255,255,255,0.08);
  --accent1:   #00f0ff;
  --accent2:   #7b5ea7;
  --accent3:   #ff4fcb;
  --text:      #e8eaf6;
  --muted:     #6b7280;
  --glow1:     0 0 30px rgba(0,240,255,0.15);
  --glow2:     0 0 60px rgba(123,94,167,0.2);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  top: -30%; left: -20%;
  width: 70vw; height: 70vw;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,240,255,0.07) 0%, transparent 70%);
  animation: drift1 12s ease-in-out infinite alternate;
  pointer-events: none; z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
  content: "";
  position: fixed;
  bottom: -20%; right: -10%;
  width: 60vw; height: 60vw;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(123,94,167,0.09) 0%, transparent 70%);
  animation: drift2 14s ease-in-out infinite alternate;
  pointer-events: none; z-index: 0;
}
@keyframes drift1 { from{transform:translate(0,0)} to{transform:translate(5%,8%)} }
@keyframes drift2 { from{transform:translate(0,0)} to{transform:translate(-6%,-5%)} }

[data-testid="block-container"] {
  position: relative; z-index: 1;
  max-width: 680px !important;
  padding-top: 2.5rem !important;
}

#MainMenu, footer, header { visibility: hidden; }

.hero { text-align: center; padding: 3rem 2rem 2rem; margin-bottom: 2rem; }
.hero-badge {
  display: inline-block;
  background: linear-gradient(135deg, rgba(0,240,255,0.12), rgba(123,94,167,0.12));
  border: 1px solid rgba(0,240,255,0.25);
  border-radius: 999px;
  padding: 0.35rem 1.1rem;
  font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--accent1); font-family: 'DM Mono', monospace; margin-bottom: 1.2rem;
}
.hero-title {
  font-size: clamp(2.6rem, 7vw, 4rem); font-weight: 800; line-height: 1.05; margin: 0 0 1rem;
  background: linear-gradient(135deg, #ffffff 0%, var(--accent1) 50%, var(--accent2) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { color: var(--muted); font-size: 0.95rem; font-family: 'DM Mono', monospace; letter-spacing: 0.02em; }

.glass-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
  padding: 2rem 2.2rem; margin-bottom: 1.4rem;
  box-shadow: var(--glow1), var(--glow2), inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter: blur(12px);
}
.card-label {
  font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--accent1); font-family: 'DM Mono', monospace; margin-bottom: 0.5rem; opacity: 0.8;
}
.section-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 1.5rem; }

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
  background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important; color: var(--text) !important;
  font-family: 'DM Mono', monospace !important; font-size: 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] select:focus {
  border-color: var(--accent1) !important; box-shadow: 0 0 0 3px rgba(0,240,255,0.12) !important; outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
  color: var(--muted) !important; font-size: 0.8rem !important;
  letter-spacing: 0.05em !important; font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] button {
  background: rgba(255,255,255,0.05) !important;
  border-color: rgba(255,255,255,0.1) !important; color: var(--accent1) !important;
}

[data-testid="stButton"] > button {
  width: 100%; padding: 0.9rem 2rem;
  font-family: 'Syne', sans-serif !important; font-weight: 700; font-size: 1rem;
  letter-spacing: 0.08em; text-transform: uppercase;
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  border: none !important; border-radius: 14px !important; color: #000 !important; cursor: pointer;
  box-shadow: 0 0 30px rgba(0,240,255,0.25), 0 4px 24px rgba(0,0,0,0.4) !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 50px rgba(0,240,255,0.4), 0 8px 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0px) !important; }

.result-card {
  border-radius: 20px; padding: 2.5rem 2rem; text-align: center; margin-top: 1.5rem;
  animation: popIn 0.45s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes popIn {
  from { opacity:0; transform:scale(0.88) translateY(10px); }
  to   { opacity:1; transform:scale(1)    translateY(0);    }
}
.result-card.yes {
  background: linear-gradient(135deg, rgba(255,79,203,0.12), rgba(123,94,167,0.15));
  border: 1px solid rgba(255,79,203,0.3); box-shadow: 0 0 60px rgba(255,79,203,0.15);
}
.result-card.no {
  background: linear-gradient(135deg, rgba(0,240,255,0.1), rgba(0,200,130,0.1));
  border: 1px solid rgba(0,240,255,0.3); box-shadow: 0 0 60px rgba(0,240,255,0.15);
}
.result-icon { font-size: 3rem; margin-bottom: 0.6rem; display: block; }
.result-label {
  font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
  font-family: 'DM Mono', monospace; opacity: 0.6; margin-bottom: 0.4rem;
}
.result-verdict { font-size: 2.4rem; font-weight: 800; line-height: 1; margin-bottom: 0.6rem; }
.result-card.yes .result-verdict { color: var(--accent3); }
.result-card.no  .result-verdict { color: var(--accent1); }
.result-desc {
  font-size: 0.82rem; color: var(--muted); font-family: 'DM Mono', monospace;
  max-width: 320px; margin: 0 auto; line-height: 1.6;
}

.conf-wrap { margin-top: 1.4rem; }
.conf-label {
  display: flex; justify-content: space-between;
  font-size: 0.7rem; font-family: 'DM Mono', monospace; color: var(--muted); margin-bottom: 0.4rem;
}
.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 999px; height: 6px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 999px; }
.conf-bar-fill.yes { background: linear-gradient(90deg, var(--accent2), var(--accent3)); }
.conf-bar-fill.no  { background: linear-gradient(90deg, var(--accent1), #00ff99); }

.analysis-header {
  font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--accent1); font-family: 'DM Mono', monospace;
  margin: 2.5rem 0 1rem; opacity: 0.8;
}
.analysis-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 1.5rem; }

.rec-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
  display: flex; gap: 1rem; align-items: flex-start;
}
.rec-card.churn { border-left: 3px solid var(--accent3); }
.rec-card.stay  { border-left: 3px solid var(--accent1); }
.rec-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
.rec-title { font-size: 0.85rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }
.rec-desc  { font-size: 0.75rem; color: var(--muted); font-family: 'DM Mono', monospace; line-height: 1.6; }

.factor-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.factor-name { font-size: 0.75rem; font-family: 'DM Mono', monospace; color: var(--muted); width: 120px; flex-shrink: 0; }
.factor-track { flex: 1; background: rgba(255,255,255,0.06); border-radius: 999px; height: 7px; overflow: hidden; }
.factor-fill  { height: 100%; border-radius: 999px; }
.factor-val   { font-size: 0.72rem; font-family: 'DM Mono', monospace; width: 40px; text-align: right; flex-shrink: 0; }

.idle-hint {
  text-align: center; padding: 1.5rem; color: var(--muted);
  font-family: 'DM Mono', monospace; font-size: 0.78rem; letter-spacing: 0.04em; margin-top: 0.5rem;
}
.idle-hint span { display: block; margin-bottom: 0.3rem; font-size: 1.6rem; opacity: 0.4; }

[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important;
}
[data-baseweb="select"] span { color: var(--text) !important; font-family: 'DM Mono', monospace !important; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.stat-chips { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1rem; }
.stat-chip {
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 10px; padding: 0.5rem 0.9rem; font-size: 0.72rem;
  font-family: 'DM Mono', monospace; color: var(--muted); flex: 1; min-width: 120px; text-align: center;
}
.stat-chip strong { display: block; color: var(--text); font-size: 1.1rem; margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_models()

# ── Plotly dark chart theme ───────────────────────────────────────────────────
PLOT_BG   = "rgba(0,0,0,0)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_COL  = "rgba(255,255,255,0.06)"
TEXT_COL  = "#6b7280"
FONT_FAM  = "DM Mono, monospace"

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#e8eaf6", family=FONT_FAM), x=0),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color=TEXT_COL, size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, showline=False),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL, showline=False),
        showlegend=False,
    )

# ── Risk scoring helpers ──────────────────────────────────────────────────────
def tenure_risk(t):
    if t <= 6:   return 92
    if t <= 12:  return 72
    if t <= 24:  return 50
    if t <= 48:  return 30
    return 14

def charge_risk(c):
    if c < 40:   return 18
    if c < 60:   return 30
    if c < 80:   return 48
    if c < 100:  return 65
    return 78

def age_risk(a):
    if a < 30:   return 35
    if a < 45:   return 28
    if a < 60:   return 50
    return 62

def gender_risk(g):
    return 20 if g == "Male" else 25

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ AI-Powered Analytics</div>
  <h1 class="hero-title">ChurnSight</h1>
  <p class="hero-sub">// predict · prevent · retain</p>
</div>
""", unsafe_allow_html=True)

# ── FORM ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">01 — Customer Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age    = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=25)
with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, max_value=500, value=65)
    gender          = st.selectbox("Gender", ["Male", "Female"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Live summary chips ────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-chips">
  <div class="stat-chip"><strong>{age}</strong>Years Old</div>
  <div class="stat-chip"><strong>{tenure}mo</strong>Customer Tenure</div>
  <div class="stat-chip"><strong>${monthly_charges}</strong>Monthly Spend</div>
  <div class="stat-chip"><strong>{'♀' if gender=='Female' else '♂'} {gender}</strong>Gender</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("⚡  Run Prediction", use_container_width=True)

# ── Result + Analysis ─────────────────────────────────────────────────────────
if predict_clicked:
    gender_enc = 1 if gender == "Female" else 0
    X_raw      = [age, gender_enc, tenure, monthly_charges]
    X_scaled   = scaler.transform([X_raw])
    prediction = model.predict(X_scaled)[0]

    try:
        proba     = model.predict_proba(X_scaled)[0]
        churn_p   = proba[1]
        retain_p  = proba[0]
        has_proba = True
    except Exception:
        churn_p  = 0.5
        retain_p = 0.5
        has_proba = False

    # ── Prediction result card ────────────────────────────────────────────────
    if prediction == 1:
        bar_w = f"{churn_p*100:.0f}%"
        st.markdown(f"""
        <div class="result-card yes">
          <span class="result-icon">⚠️</span>
          <div class="result-label">Churn Prediction</div>
          <div class="result-verdict">LIKELY TO CHURN</div>
          <p class="result-desc">
            This customer shows a high risk of leaving.
            Consider proactive retention strategies — personalized offers,
            loyalty rewards, or direct outreach.
          </p>
          {'<div class="conf-wrap"><div class="conf-label"><span>Churn probability</span><span>' + f"{churn_p*100:.1f}%" + '</span></div><div class="conf-bar-bg"><div class="conf-bar-fill yes" style="width:' + bar_w + '"></div></div></div>' if has_proba else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        bar_w = f"{retain_p*100:.0f}%"
        st.markdown(f"""
        <div class="result-card no">
          <span class="result-icon">✅</span>
          <div class="result-label">Churn Prediction</div>
          <div class="result-verdict">LIKELY TO STAY</div>
          <p class="result-desc">
            This customer appears stable and satisfied.
            Keep up engagement through regular check-ins
            and value-driven communication.
          </p>
          {'<div class="conf-wrap"><div class="conf-label"><span>Retention probability</span><span>' + f"{retain_p*100:.1f}%" + '</span></div><div class="conf-bar-bg"><div class="conf-bar-fill no" style="width:' + bar_w + '"></div></div></div>' if has_proba else ''}
        </div>
        """, unsafe_allow_html=True)

    # ── Compute per-factor risk scores ────────────────────────────────────────
    t_risk = tenure_risk(tenure)
    c_risk = charge_risk(monthly_charges)
    a_risk = age_risk(age)
    g_risk = gender_risk(gender)

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="analysis-header">02 — Factor Analysis</div>
    <div class="analysis-title">Why this prediction?</div>
    """, unsafe_allow_html=True)

    # ── Risk factor bar breakdown ─────────────────────────────────────────────
    def risk_color(score):
        if score >= 65: return "#ff4fcb"
        if score >= 40: return "#f59e0b"
        return "#00f0ff"

    def risk_label(score):
        if score >= 65: return "HIGH"
        if score >= 40: return "MED"
        return "LOW"

    st.markdown(f"""
    <div class="glass-card">
      <div class="card-label">Risk factor breakdown</div>

      <div class="factor-row">
        <span class="factor-name">Tenure</span>
        <div class="factor-track"><div class="factor-fill" style="width:{t_risk}%; background:{risk_color(t_risk)};"></div></div>
        <span class="factor-val" style="color:{risk_color(t_risk)};">{risk_label(t_risk)}</span>
      </div>

      <div class="factor-row">
        <span class="factor-name">Monthly Charges</span>
        <div class="factor-track"><div class="factor-fill" style="width:{c_risk}%; background:{risk_color(c_risk)};"></div></div>
        <span class="factor-val" style="color:{risk_color(c_risk)};">{risk_label(c_risk)}</span>
      </div>

      <div class="factor-row">
        <span class="factor-name">Age</span>
        <div class="factor-track"><div class="factor-fill" style="width:{a_risk}%; background:{risk_color(a_risk)};"></div></div>
        <span class="factor-val" style="color:{risk_color(a_risk)};">{risk_label(a_risk)}</span>
      </div>

      <div class="factor-row">
        <span class="factor-name">Gender</span>
        <div class="factor-track"><div class="factor-fill" style="width:{g_risk}%; background:{risk_color(g_risk)};"></div></div>
        <span class="factor-val" style="color:{risk_color(g_risk)};">{risk_label(g_risk)}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        # Churn rate by tenure group — bar chart
        tenure_labels = ["0–6 mo", "7–12 mo", "13–24 mo", "25–48 mo", "48+ mo"]
        tenure_rates  = [92, 72, 50, 30, 14]
        bar_colors    = ["#ff4fcb","#f59e0b","#f59e0b","#00f0ff","#00f0ff"]

        # Highlight which bucket this customer is in
        if tenure <= 6:    bucket = 0
        elif tenure <= 12: bucket = 1
        elif tenure <= 24: bucket = 2
        elif tenure <= 48: bucket = 3
        else:              bucket = 4

        fig1 = go.Figure(go.Bar(
            x=tenure_labels,
            y=tenure_rates,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v}%" for v in tenure_rates],
            textposition="outside",
            textfont=dict(size=10, color="#e8eaf6"),
        ))
        fig1.add_annotation(
            x=tenure_labels[bucket], y=tenure_rates[bucket],
            text="▲ You", showarrow=False, yshift=28,
            font=dict(size=10, color="#ffffff", family=FONT_FAM)
        )
        layout1 = base_layout("Churn rate by tenure")
        layout1["yaxis"]["ticksuffix"] = "%"
        layout1["yaxis"]["range"] = [0, 115]
        fig1.update_layout(**layout1)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        # Churn rate by monthly charge band — bar chart
        charge_labels = ["<$40","$40–60","$60–80","$80–100",">$100"]
        charge_rates  = [18, 30, 48, 65, 78]
        bar_colors2   = ["#00f0ff","#00f0ff","#f59e0b","#ff4fcb","#ff4fcb"]

        if monthly_charges < 40:    cbucket = 0
        elif monthly_charges < 60:  cbucket = 1
        elif monthly_charges < 80:  cbucket = 2
        elif monthly_charges < 100: cbucket = 3
        else:                       cbucket = 4

        fig2 = go.Figure(go.Bar(
            x=charge_labels,
            y=charge_rates,
            marker_color=bar_colors2,
            marker_line_width=0,
            text=[f"{v}%" for v in charge_rates],
            textposition="outside",
            textfont=dict(size=10, color="#e8eaf6"),
        ))
        fig2.add_annotation(
            x=charge_labels[cbucket], y=charge_rates[cbucket],
            text="▲ You", showarrow=False, yshift=28,
            font=dict(size=10, color="#ffffff", family=FONT_FAM)
        )
        layout2 = base_layout("Churn rate by monthly charges")
        layout2["yaxis"]["ticksuffix"] = "%"
        layout2["yaxis"]["range"] = [0, 100]
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Radar chart — this customer vs avg churner ────────────────────────────
    categories   = ["Tenure risk", "Charge risk", "Age risk", "Gender risk", "Overall"]
    overall_risk = int((t_risk + c_risk + a_risk + g_risk) / 4)
    customer_vals = [t_risk, c_risk, a_risk, g_risk, overall_risk]
    avg_churner   = [65, 70, 50, 28, 60]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(
        r=customer_vals + [customer_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255,79,203,0.10)",
        line=dict(color="#ff4fcb", width=2),
        name="This customer",
    ))
    fig3.add_trace(go.Scatterpolar(
        r=avg_churner + [avg_churner[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(107,114,128,0.08)",
        line=dict(color="#6b7280", width=1.5, dash="dot"),
        name="Avg churner",
    ))
    fig3.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,100], ticksuffix="%", tickfont=dict(size=9, color="#6b7280"), gridcolor=GRID_COL, linecolor=GRID_COL),
            angularaxis=dict(tickfont=dict(size=10, color="#e8eaf6", family=FONT_FAM), gridcolor=GRID_COL, linecolor=GRID_COL),
        ),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAM),
        showlegend=True,
        legend=dict(font=dict(color="#e8eaf6", size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=30, t=50, b=20),
        title=dict(text="Customer vs avg churner profile", font=dict(size=13, color="#e8eaf6", family=FONT_FAM), x=0),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="analysis-header">03 — Action Plan</div>
    <div class="analysis-title">Recommendations</div>
    """, unsafe_allow_html=True)

    card_cls = "churn" if prediction == 1 else "stay"

    if prediction == 1:
        # Dynamic recommendations based on risk factors
        recs = []

        if t_risk >= 65:
            recs.append(("🚨", "Immediate outreach required",
                f"Customers with {tenure} months tenure sit in the highest-risk window. "
                "Assign a dedicated account manager and initiate contact within 48 hours."))

        if c_risk >= 48:
            recs.append(("💰", "Offer a personalised discount",
                f"At ${monthly_charges}/mo this customer pays above the safe retention threshold. "
                "A 10–15% loyalty discount or plan downgrade option can significantly reduce churn risk."))

        if a_risk >= 50:
            recs.append(("👤", "Tailor communication for age group",
                f"Customers aged {age} show elevated churn. Offer simplified plans, "
                "dedicated phone support, or a senior-friendly interface to improve satisfaction."))

        recs.append(("🎁", "Activate loyalty programme",
            "Enrol this customer in a rewards programme immediately. "
            "Points, exclusive perks, or early access to features increase emotional stickiness."))

        recs.append(("📞", "Schedule a satisfaction check-in",
            "A proactive call asking about experience — not to sell — builds trust and "
            "surfaces hidden issues before they become churn triggers."))

        for icon, title, desc in recs:
            st.markdown(f"""
            <div class="rec-card churn">
              <span class="rec-icon">{icon}</span>
              <div>
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Retention maintenance recommendations
        recs = [
            ("✅", "Maintain current engagement cadence",
             "This customer is stable. Continue regular value-driven communication — "
             "product updates, usage tips, and milestone acknowledgements."),
            ("🌟", "Enrol in loyalty / ambassador programme",
             "Satisfied long-tenure customers are your best advocates. "
             "Invite them to referral programmes or beta features to deepen commitment."),
            ("📊", "Monitor for warning signals",
             "Even retained customers can shift. Set an alert if monthly charges increase >15% "
             "or if tenure crosses a new risk threshold to act early."),
            ("🎯", "Upsell strategically",
             "Low churn risk creates a safe window for upselling. "
             "Recommend complementary services or plan upgrades that add genuine value."),
        ]
        for icon, title, desc in recs:
            st.markdown(f"""
            <div class="rec-card stay">
              <span class="rec-icon">{icon}</span>
              <div>
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="idle-hint">
      <span>🔮</span>
      Fill in the customer details above and hit <strong>Run Prediction</strong>
    </div>
    """, unsafe_allow_html=True)