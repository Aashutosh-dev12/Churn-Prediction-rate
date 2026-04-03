import streamlit as st
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight · Prediction",
    page_icon="⚡",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
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

/* ── Full-page dark background + noise ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'Syne', sans-serif !important;
  color: var(--text) !important;
}

/* Animated gradient orbs */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  top: -30%;
  left: -20%;
  width: 70vw;
  height: 70vw;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,240,255,0.07) 0%, transparent 70%);
  animation: drift1 12s ease-in-out infinite alternate;
  pointer-events: none;
  z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
  content: "";
  position: fixed;
  bottom: -20%;
  right: -10%;
  width: 60vw;
  height: 60vw;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(123,94,167,0.09) 0%, transparent 70%);
  animation: drift2 14s ease-in-out infinite alternate;
  pointer-events: none;
  z-index: 0;
}
@keyframes drift1 { from{transform:translate(0,0)} to{transform:translate(5%,8%)} }
@keyframes drift2 { from{transform:translate(0,0)} to{transform:translate(-6%,-5%)} }

/* ── Main block ── */
[data-testid="block-container"] {
  position: relative;
  z-index: 1;
  max-width: 680px !important;
  padding-top: 2.5rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero {
  text-align: center;
  padding: 3rem 2rem 2rem;
  margin-bottom: 2rem;
}
.hero-badge {
  display: inline-block;
  background: linear-gradient(135deg, rgba(0,240,255,0.12), rgba(123,94,167,0.12));
  border: 1px solid rgba(0,240,255,0.25);
  border-radius: 999px;
  padding: 0.35rem 1.1rem;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent1);
  font-family: 'DM Mono', monospace;
  margin-bottom: 1.2rem;
}
.hero-title {
  font-size: clamp(2.6rem, 7vw, 4rem);
  font-weight: 800;
  line-height: 1.05;
  margin: 0 0 1rem;
  background: linear-gradient(135deg, #ffffff 0%, var(--accent1) 50%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  color: var(--muted);
  font-size: 0.95rem;
  font-family: 'DM Mono', monospace;
  letter-spacing: 0.02em;
}

/* ── Glass card ── */
.glass-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.4rem;
  box-shadow: var(--glow1), var(--glow2), inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter: blur(12px);
}
.card-label {
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent1);
  font-family: 'DM Mono', monospace;
  margin-bottom: 0.5rem;
  opacity: 0.8;
}
.section-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 1.5rem;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: 1rem !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] select:focus {
  border-color: var(--accent1) !important;
  box-shadow: 0 0 0 3px rgba(0,240,255,0.12) !important;
  outline: none !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
  color: var(--muted) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.05em !important;
  font-family: 'DM Mono', monospace !important;
}

/* Number input stepper buttons */
[data-testid="stNumberInput"] button {
  background: rgba(255,255,255,0.05) !important;
  border-color: rgba(255,255,255,0.1) !important;
  color: var(--accent1) !important;
}

/* ── Gender pills ── */
.gender-row {
  display: flex;
  gap: 1rem;
  margin-top: 0.3rem;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
  width: 100%;
  padding: 0.9rem 2rem;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  border: none !important;
  border-radius: 14px !important;
  color: #000 !important;
  cursor: pointer;
  box-shadow: 0 0 30px rgba(0,240,255,0.25), 0 4px 24px rgba(0,0,0,0.4) !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 50px rgba(0,240,255,0.4), 0 8px 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stButton"] > button:active {
  transform: translateY(0px) !important;
}

/* ── Result cards ── */
.result-card {
  border-radius: 20px;
  padding: 2.5rem 2rem;
  text-align: center;
  margin-top: 1.5rem;
  animation: popIn 0.45s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes popIn {
  from { opacity:0; transform:scale(0.88) translateY(10px); }
  to   { opacity:1; transform:scale(1)    translateY(0);    }
}
.result-card.yes {
  background: linear-gradient(135deg, rgba(255,79,203,0.12), rgba(123,94,167,0.15));
  border: 1px solid rgba(255,79,203,0.3);
  box-shadow: 0 0 60px rgba(255,79,203,0.15);
}
.result-card.no {
  background: linear-gradient(135deg, rgba(0,240,255,0.1), rgba(0,200,130,0.1));
  border: 1px solid rgba(0,240,255,0.3);
  box-shadow: 0 0 60px rgba(0,240,255,0.15);
}
.result-icon {
  font-size: 3rem;
  margin-bottom: 0.6rem;
  display: block;
  filter: drop-shadow(0 0 12px currentColor);
}
.result-label {
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  font-family: 'DM Mono', monospace;
  opacity: 0.6;
  margin-bottom: 0.4rem;
}
.result-verdict {
  font-size: 2.4rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 0.6rem;
}
.result-card.yes .result-verdict { color: var(--accent3); }
.result-card.no  .result-verdict { color: var(--accent1); }
.result-desc {
  font-size: 0.82rem;
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  max-width: 320px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ── Confidence bar ── */
.conf-wrap {
  margin-top: 1.4rem;
}
.conf-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  font-family: 'DM Mono', monospace;
  color: var(--muted);
  margin-bottom: 0.4rem;
}
.conf-bar-bg {
  background: rgba(255,255,255,0.06);
  border-radius: 999px;
  height: 6px;
  overflow: hidden;
}
.conf-bar-fill {
  height: 100%;
  border-radius: 999px;
  animation: growBar 0.8s cubic-bezier(0.22,1,0.36,1) both;
  animation-delay: 0.25s;
}
.conf-bar-fill.yes { background: linear-gradient(90deg, var(--accent2), var(--accent3)); }
.conf-bar-fill.no  { background: linear-gradient(90deg, var(--accent1), #00ff99); }
@keyframes growBar {
  from { width: 0%; }
  to   { width: var(--bar-w); }
}

/* ── Idle hint ── */
.idle-hint {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted);
  font-family: 'DM Mono', monospace;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  margin-top: 0.5rem;
}
.idle-hint span { display: block; margin-bottom: 0.3rem; font-size: 1.6rem; opacity: 0.4; }

/* ── Selectbox dropdown fix ── */
[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 10px !important;
}
[data-baseweb="select"] span {
  color: var(--text) !important;
  font-family: 'DM Mono', monospace !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Stat chips inside card ── */
.stat-chips {
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}
.stat-chip {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 10px;
  padding: 0.5rem 0.9rem;
  font-size: 0.72rem;
  font-family: 'DM Mono', monospace;
  color: var(--muted);
  flex: 1;
  min-width: 120px;
  text-align: center;
}
.stat-chip strong {
  display: block;
  color: var(--text);
  font-size: 1.1rem;
  margin-bottom: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model  = joblib.load("model.pkl")
    return scaler, model

scaler, model = load_models()

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
    age     = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure  = st.number_input("Tenure (months)", min_value=0, max_value=120, value=25)
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

# ── Result ────────────────────────────────────────────────────────────────────
if predict_clicked:
    gender_enc = 1 if gender == "Female" else 0
    X          = [age, gender_enc, tenure, monthly_charges]
    X_scaled   = scaler.transform([X])
    prediction = model.predict(X_scaled)[0]

    # Try to get probability
    try:
        proba     = model.predict_proba(X_scaled)[0]
        churn_p   = proba[1]          # probability of churn
        retain_p  = proba[0]
        has_proba = True
    except Exception:
        churn_p  = 0.5
        retain_p = 0.5
        has_proba = False

    if prediction == 1:
        bar_w   = f"{churn_p*100:.0f}%"
        bar_cls = "yes"
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
          {'<div class="conf-wrap"><div class="conf-label"><span>Churn probability</span><span>' + f"{churn_p*100:.1f}%" + '</span></div><div class="conf-bar-bg"><div class="conf-bar-fill yes" style="--bar-w:' + bar_w + '; width:' + bar_w + '"></div></div></div>' if has_proba else ''}
        </div>
        """, unsafe_allow_html=True)
    else:
        bar_w   = f"{retain_p*100:.0f}%"
        bar_cls = "no"
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
          {'<div class="conf-wrap"><div class="conf-label"><span>Retention probability</span><span>' + f"{retain_p*100:.1f}%" + '</span></div><div class="conf-bar-bg"><div class="conf-bar-fill no" style="--bar-w:' + bar_w + '; width:' + bar_w + '"></div></div></div>' if has_proba else ''}
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="idle-hint">
      <span>🔮</span>
      Fill in the customer details above and hit <strong>Run Prediction</strong>
    </div>
    """, unsafe_allow_html=True)