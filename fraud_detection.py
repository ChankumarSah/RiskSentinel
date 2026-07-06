import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="RiskSentinel | Fraud Scoring",
    page_icon="🛡️",
    layout="centered"
)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("Fraud_Detection_pipeline.pkl")
# model = joblib.load("FraudDetection_RF_SMOTE.pkl")

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

if "prefill" not in st.session_state:
    st.session_state.prefill = None

# ===============================
# CUSTOM CSS — PREMIUM GLASSMORPHISM THEME
# ===============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated gradient background */
.stApp{
    background: linear-gradient(-45deg, #05060f, #0f172a, #1e1b4b, #111827, #1e293b);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    color:white;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Soft glowing orbs behind content */
.stApp::before{
    content:"";
    position:fixed;
    top:-150px;
    left:-150px;
    width:400px;
    height:400px;
    background:radial-gradient(circle, rgba(124,58,237,0.35), transparent 70%);
    border-radius:50%;
    z-index:0;
    pointer-events:none;
}

.stApp::after{
    content:"";
    position:fixed;
    bottom:-180px;
    right:-180px;
    width:450px;
    height:450px;
    background:radial-gradient(circle, rgba(37,99,235,0.30), transparent 70%);
    border-radius:50%;
    z-index:0;
    pointer-events:none;
}

/* Hide Streamlit chrome */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main content container - glass card feel */
.block-container{
    position:relative;
    z-index:1;
    max-width:780px;
    padding-top:2.2rem;
}

/* Title */
.title{
    text-align:center;
    font-family:'Poppins', sans-serif;
    font-size:46px;
    font-weight:800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    letter-spacing:0.5px;
    margin-bottom:6px;
}

.subtitle{
    text-align:center;
    color:#94A3B8;
    font-size:16px;
    font-weight:500;
    letter-spacing:0.3px;
    margin-bottom:30px;
}

/* Glass panel wrapper for sections */
.glass-card{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 26px 28px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.section-heading{
    font-family:'Poppins', sans-serif;
    font-size:20px;
    font-weight:700;
    color:#E2E8F0;
    margin-bottom:14px;
    display:flex;
    align-items:center;
    gap:8px;
}

/* Streamlit's own subheader / divider polish */
hr{
    border-color: rgba(255,255,255,0.08) !important;
}

/* Info box */
div[data-testid="stAlertContainer"]{
    background: rgba(37,99,235,0.12) !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
}

/* Labels */
label, .stNumberInput label, .stSelectbox label{
    color:#CBD5E1 !important;
    font-weight:600 !important;
    font-size:14px !important;
}

/* Input boxes */
input, .stNumberInput input{
    background: rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    border-radius:12px !important;
    color:white !important;
}

input:focus{
    border:1px solid #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.35) !important;
}

div[data-baseweb="select"] > div{
    background: rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    border-radius:12px !important;
    color:white !important;
}

/* Button */
.stButton>button{
    width:100%;
    height:58px;
    border-radius:14px;
    border:none;
    background:linear-gradient(90deg,#2563eb,#7c3aed,#db2777);
    background-size:200% auto;
    color:white;
    font-size:19px;
    font-weight:700;
    letter-spacing:0.4px;
    box-shadow: 0 8px 24px rgba(124,58,237,0.35);
    transition: all 0.35s ease;
}

.stButton>button:hover{
    background-position: right center;
    transform:translateY(-2px) scale(1.01);
    box-shadow: 0 12px 30px rgba(219,39,119,0.4);
}

.stButton>button:active{
    transform:translateY(0px) scale(0.99);
}

/* Metric styling */
div[data-testid="stMetric"]{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 14px 18px;
}

div[data-testid="stMetricValue"]{
    color:#E2E8F0;
    font-family:'Poppins', sans-serif;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div{
    background: linear-gradient(90deg,#2563eb,#7c3aed,#db2777) !important;
    border-radius: 10px;
}

/* Result cards */
.result-card{
    border-radius:20px;
    padding:26px 28px;
    text-align:center;
    backdrop-filter: blur(16px);
    margin-top:10px;
}

.result-fraud{
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(248,113,113,0.45);
    box-shadow: 0 0 40px rgba(239,68,68,0.15);
}

.result-safe{
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(74,222,128,0.45);
    box-shadow: 0 0 40px rgba(34,197,94,0.15);
}

.result-title{
    font-family:'Poppins', sans-serif;
    font-size:22px;
    font-weight:700;
    margin-bottom:6px;
}

.result-percent{
    font-family:'Poppins', sans-serif;
    font-size:44px;
    font-weight:800;
    margin:6px 0;
}

.result-sub{
    color:#CBD5E1;
    font-size:15px;
}

/* Footer */
.footer{
    text-align:center;
    color:#64748B;
    font-size:13px;
    margin-top:20px;
    padding-bottom:10px;
}

/* Badges row */
.badge-row{
    display:flex;
    justify-content:center;
    gap:10px;
    flex-wrap:wrap;
    margin-bottom:26px;
}

.badge{
    background: rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.14);
    padding:6px 14px;
    border-radius:999px;
    font-size:12.5px;
    font-weight:600;
    color:#CBD5E1;
    backdrop-filter: blur(8px);
}

/* Balance change mini-cards */
.mini-card{
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:14px;
    padding:12px 14px;
    text-align:center;
}

.mini-card .label{
    font-size:12px;
    color:#94A3B8;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.5px;
}

.mini-card .value{
    font-size:20px;
    font-weight:700;
    margin-top:4px;
}

/* Sidebar polish */
section[data-testid="stSidebar"]{
    background: rgba(15,23,42,0.85);
    border-right:1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] .stMarkdown p, 
section[data-testid="stSidebar"] li{
    color:#CBD5E1 !important;
    font-size:14px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap:6px;
    background: rgba(255,255,255,0.04);
    padding:6px;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.08);
}

.stTabs [data-baseweb="tab"]{
    border-radius:10px;
    color:#94A3B8;
    font-weight:600;
    padding:8px 18px;
}

.stTabs [aria-selected="true"]{
    background: linear-gradient(90deg,#2563eb,#7c3aed) !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# SAMPLE DATA
# ===============================
FRAUD_SAMPLE = {
    "type": "TRANSFER", "amount": 181000.0,
    "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0, "newbalanceDest": 0.0
}
SAFE_SAMPLE = {
    "type": "PAYMENT", "amount": 950.0,
    "oldbalanceOrg": 15000.0, "newbalanceOrig": 14050.0,
    "oldbalanceDest": 5000.0, "newbalanceDest": 5950.0
}

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### 🛡️ RiskSentinel")
    st.caption("A Machine Learning Approach to Real-Time Fraud Scoring")
    st.divider()

    st.markdown("**⚙️ How it works**")
    st.markdown("""
    1. Enter transaction details  
    2. Model engineers balance-change features  
    3. Classifier scores fraud probability  
    4. Get an instant risk verdict  
    """)

    st.divider()
    st.markdown("**📈 Model Stack**")
    st.markdown("- Scikit-Learn Pipeline\n- Balance-diff feature engineering\n- Probability-calibrated classifier")

    st.divider()
    st.markdown("**🕓 Recent Predictions**")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[-5:][::-1])
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No predictions yet this session.")

    st.divider()
    st.markdown("**👨‍💻 Developer**")
    st.markdown("""
    <div style="line-height:2;">
    <b>Chandan Kumar Sah</b><br>
    <span style="color:#94A3B8;font-size:13px;">Trained &amp; developed this model</span><br>
    🔗 <a href="https://www.linkedin.com/in/chandan-kumar-sah-752803387/" target="_blank" style="color:#60a5fa;text-decoration:none;">LinkedIn</a>
    &nbsp;·&nbsp;
    💻 <a href="https://github.com/ChankumarSah" target="_blank" style="color:#a78bfa;text-decoration:none;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================

st.markdown("""
<div class='title'>
🛡️ RiskSentinel
</div>

<div class='subtitle'>
A Machine Learning Approach to Real-Time Fraud Scoring
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <div class="badge">🔒 Secure & Private</div>
    <div class="badge">⚡ Real-time Scoring</div>
    <div class="badge">🤖 ML Powered</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(["🔍 Predict", "ℹ️ About"])

with tab_predict:

    st.info("Fill in the transaction details below and click **Predict Transaction** to get an instant risk assessment.")

    qcol1, qcol2 = st.columns(2)
    with qcol1:
        if st.button("🧪 Try Fraud Example"):
            st.session_state.prefill = FRAUD_SAMPLE
    with qcol2:
        if st.button("✅ Try Safe Example"):
            st.session_state.prefill = SAFE_SAMPLE

    pf = st.session_state.prefill

    # ===============================
    # INPUTS
    # ===============================

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">💳 Transaction Details</div>', unsafe_allow_html=True)

    type_options = ["PAYMENT","TRANSFER","CASH_OUT","DEPOSIT"]
    transaction_type = st.selectbox(
        "Transaction Type",
        type_options,
        index=type_options.index(pf["type"]) if pf else 0
    )

    col1,col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Amount",
            min_value=0.0,
            value=pf["amount"] if pf else 1000.0
        )

        oldbalanceOrg = st.number_input(
            "Old Balance (Sender)",
            min_value=0.0,
            value=pf["oldbalanceOrg"] if pf else 10000.0
        )

        oldbalanceDest = st.number_input(
            "Old Balance (Receiver)",
            min_value=0.0,
            value=pf["oldbalanceDest"] if pf else 0.0
        )

    with col2:

        newbalanceOrig = st.number_input(
            "New Balance (Sender)",
            min_value=0.0,
            value=pf["newbalanceOrig"] if pf else 9000.0
        )

        newbalanceDest = st.number_input(
            "New Balance (Receiver)",
            min_value=0.0,
            value=pf["newbalanceDest"] if pf else 0.0
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================

    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = oldbalanceDest - newbalanceDest

    # Balance change summary cards
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📉 Balance Change Summary</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    orig_arrow = "🔻" if balanceDiffOrig > 0 else ("🔺" if balanceDiffOrig < 0 else "➖")
    dest_arrow = "🔻" if balanceDiffDest > 0 else ("🔺" if balanceDiffDest < 0 else "➖")

    with mc1:
        st.markdown(f"""
        <div class="mini-card">
            <div class="label">Sender Balance Change</div>
            <div class="value">{orig_arrow} {abs(balanceDiffOrig):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="mini-card">
            <div class="label">Receiver Balance Change</div>
            <div class="value">{dest_arrow} {abs(balanceDiffDest):,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # PREDICTION
    # ===============================

    predict_clicked = st.button("🚀 Predict Transaction")

    if predict_clicked:

        input_data = pd.DataFrame([{

            "type": transaction_type,

            "amount": amount,

            "oldbalanceOrg": oldbalanceOrg,

            "newbalanceOrig": newbalanceOrig,

            "oldbalanceDest": oldbalanceDest,

            "newbalanceDest": newbalanceDest,

            "balanceDiffOrig": balanceDiffOrig,

            "balanceDiffDest": balanceDiffDest

        }])

        with st.spinner("Analyzing transaction pattern..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

        confidence = probability[int(prediction)] * 100
        fraud_pct = probability[1] * 100

        # Save to session history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Type": transaction_type,
            "Amount": amount,
            "Verdict": "FRAUD" if prediction == 1 else "SAFE",
            "Confidence": f"{confidence:.1f}%"
        })

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📊 Prediction Result</div>', unsafe_allow_html=True)

        # Risk gauge chart
        gauge_color = "#f87171" if prediction == 1 else "#4ade80"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_pct,
            number={'suffix': "%", 'font': {'color': 'white', 'size': 40}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#94A3B8", 'tickfont': {'color': '#94A3B8'}},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(255,255,255,0.03)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.15)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(74,222,128,0.15)'},
                    {'range': [40, 70], 'color': 'rgba(251,191,36,0.15)'},
                    {'range': [70, 100], 'color': 'rgba(248,113,113,0.15)'}
                ],
            },
            title={'text': "Fraud Risk Score", 'font': {'color': '#CBD5E1', 'size': 16}}
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=50, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Model Confidence", f"{confidence:.2f}%")

        if prediction == 1:

            st.markdown(f"""
            <div class="result-card result-fraud">
                <div class="result-title">🚨 FRAUD DETECTED</div>
                <div class="result-percent" style="color:#f87171;">{probability[1]*100:.2f}%</div>
                <div class="result-sub">Fraud probability — please verify this transaction before approving it.</div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card result-safe">
                <div class="result-title">✅ LEGITIMATE TRANSACTION</div>
                <div class="result-percent" style="color:#4ade80;">{probability[0]*100:.2f}%</div>
                <div class="result-sub">Legitimate probability — no suspicious activity detected.</div>
            </div>
            """, unsafe_allow_html=True)

        report_text = f"""RISKSENTINEL — FRAUD DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
------------------------------------------
Transaction Type : {transaction_type}
Amount           : {amount:,.2f}
Old Balance (Sender)   : {oldbalanceOrg:,.2f}
New Balance (Sender)   : {newbalanceOrig:,.2f}
Old Balance (Receiver) : {oldbalanceDest:,.2f}
New Balance (Receiver) : {newbalanceDest:,.2f}
------------------------------------------
Verdict          : {"FRAUD DETECTED" if prediction == 1 else "LEGITIMATE"}
Fraud Probability: {probability[1]*100:.2f}%
Model Confidence : {confidence:.2f}%
"""
        st.download_button(
            "📄 Download Report",
            data=report_text,
            file_name=f"risksentinel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

        st.markdown('</div>', unsafe_allow_html=True)

with tab_about:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">ℹ️ About RiskSentinel</div>', unsafe_allow_html=True)
    st.markdown("""
    **RiskSentinel** is a machine learning approach to real-time fraud scoring —
    it uses a trained ML pipeline to flag potentially fraudulent financial
    transactions as they happen.

    **Features used by the model:**
    - Transaction type (Payment, Transfer, Cash-Out, Deposit)
    - Transaction amount
    - Sender's old & new balance
    - Receiver's old & new balance
    - Engineered balance-change features

    **Note:** This is a decision-support tool. High-risk flags should
    always be reviewed by a human analyst before final action.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================

st.markdown("""
<div class="glass-card" style="text-align:center; padding:20px 24px;">
    <div style="font-family:'Poppins', sans-serif; font-weight:700; font-size:16px; color:#E2E8F0;">
        👨‍💻 Model Trained &amp; Developed by
    </div>
    <div style="font-family:'Poppins', sans-serif; font-weight:800; font-size:20px; margin:6px 0; background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
        Chandan Kumar Sah
    </div>
    <div style="margin-top:10px;">
        <a href="https://www.linkedin.com/in/chandan-kumar-sah-752803387/" target="_blank" style="text-decoration:none;">
            <span class="badge">🔗 LinkedIn</span>
        </a>
        &nbsp;
        <a href="https://github.com/ChankumarSah" target="_blank" style="text-decoration:none;">
            <span class="badge">💻 GitHub</span>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class='footer'>
Developed with ❤️ &nbsp;|&nbsp; <b>Python · Scikit-Learn · Streamlit · Plotly · Machine Learning</b>
</div>
""",
unsafe_allow_html=True
)