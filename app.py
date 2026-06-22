import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Workforce Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (sidebar toggle CSS removed — was breaking the button)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0a0f; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stFileUploader label {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #6b7db3 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #4f6ef7 !important;
    border: 2px solid #7b93ff !important;
    box-shadow: 0 0 10px rgba(79,110,247,0.5) !important;
}

/* ── Expander styling ── */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: #13131f !important;
    border: 1px solid #1e1e3a !important;
    border-radius: 8px !important;
    color: #c7d2f0 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 0.75rem !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    background: #1a1a2e !important;
    border-color: #4f6ef7 !important;
}
[data-testid="stSidebar"] .streamlit-expanderContent {
    background: #0f0f1a !important;
    border: 1px solid #1e1e3a !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 0.6rem 0.5rem !important;
}

/* ── Sidebar header ── */
.sidebar-brand {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #4f6ef7;
    font-family: 'JetBrains Mono', monospace;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1e1e3a;
    margin-bottom: 1rem;
}

/* ── Main header ── */
.main-header {
    padding: 2.2rem 0 1.8rem 0;
    border-bottom: 1px solid #1e1e3a;
    margin-bottom: 1.8rem;
}
.main-header .eyebrow {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: #4f6ef7;
    font-family: 'JetBrains Mono', monospace;
}
.main-header h1 {
    font-size: 2rem; font-weight: 700; color: #f0f4ff;
    line-height: 1.15; margin: 0.3rem 0; letter-spacing: -0.02em;
}
.main-header .sub { font-size: 0.88rem; color: #4a5578; }

/* ── Section headers ── */
.section-header {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #4f6ef7;
    font-family: 'JetBrains Mono', monospace;
    margin: 1.6rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a2e;
}

/* ── Risk cards ── */
.risk-card {
    border-radius: 12px; padding: 1.4rem 1.6rem;
    display: flex; flex-direction: column; gap: 0.35rem;
    border: 1px solid transparent;
}
.risk-high   { background: linear-gradient(135deg,#1f0a0a,#2d0f0f); border-color:#7f1d1d; }
.risk-medium { background: linear-gradient(135deg,#1a150a,#2a1f0d); border-color:#78350f; }
.risk-low    { background: linear-gradient(135deg,#051a10,#0a2418); border-color:#14532d; }
.risk-label  { font-size:0.62rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; font-family:'JetBrains Mono',monospace; }
.risk-high   .risk-label { color:#f87171; }
.risk-medium .risk-label { color:#fbbf24; }
.risk-low    .risk-label { color:#34d399; }
.risk-value  { font-size:2.6rem; font-weight:700; line-height:1; letter-spacing:-0.03em; }
.risk-high   .risk-value { color:#ef4444; }
.risk-medium .risk-value { color:#f59e0b; }
.risk-low    .risk-value { color:#10b981; }
.risk-desc   { font-size:0.8rem; color:#6b7db3; margin-top:0.1rem; }

/* ── Metric cards ── */
.metric-card {
    background:#0f0f1a; border:1px solid #1e1e3a; border-radius:12px;
    padding:1.3rem 1.5rem; display:flex; flex-direction:column; gap:0.3rem;
}
.metric-label { font-size:0.62rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:#3d4f7c; font-family:'JetBrains Mono',monospace; }
.metric-value { font-size:2rem; font-weight:700; color:#c7d2f0; letter-spacing:-0.02em; line-height:1.1; }
.metric-sub   { font-size:0.75rem; color:#3d4f7c; }

/* ── Dataset badge ── */
.dataset-badge {
    display:inline-flex; align-items:center; gap:0.5rem;
    background:#0f0f1a; border:1px solid #1e1e3a; border-radius:8px;
    padding:0.55rem 1rem; font-size:0.75rem; color:#6b7db3;
    font-family:'JetBrains Mono',monospace; margin-bottom:1.6rem;
}
.dataset-badge .dot {
    width:7px; height:7px; border-radius:50%; background:#4f6ef7;
    box-shadow:0 0 6px rgba(79,110,247,0.8); display:inline-block;
}

/* ── Predict button ── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg,#4f6ef7,#7b5cf0);
    color:#fff; font-family:'Space Grotesk',sans-serif;
    font-weight:600; font-size:0.85rem; letter-spacing:0.05em;
    border:none; border-radius:10px; padding:0.75rem 1.5rem;
    width:100%; cursor:pointer; transition:all 0.2s;
    box-shadow:0 4px 20px rgba(79,110,247,0.3); margin-top:0.4rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow:0 6px 28px rgba(79,110,247,0.5); transform:translateY(-1px);
}

/* ── Misc ── */
hr { border-color:#1e1e3a !important; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header { visibility:hidden; }
[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
section[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; display: flex !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0a0f; }
::-webkit-scrollbar-thumb { background:#1e1e3a; border-radius:10px; }
[data-testid="stNumberInput"] input {
    background:#13131f !important; border:1px solid #1e1e3a !important;
    color:#e2e8f0 !important; border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important;
}
[data-testid="stSelectbox"] > div > div {
    background:#13131f !important; border:1px solid #1e1e3a !important;
    border-radius:8px !important; color:#e2e8f0 !important;
}
[data-testid="stFileUploader"] {
    background:#0f0f1a; border:1px dashed #1e1e3a; border-radius:10px; padding:0.4rem;
}
.stAlert { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#6b7db3", size=12),
    title_font=dict(family="Space Grotesk", color="#c7d2f0", size=14),
    margin=dict(l=16, r=16, t=48, b=16),
    colorway=["#4f6ef7","#7b5cf0","#34d399","#f59e0b","#f87171","#38bdf8"],
)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load("attrition_model.pkl")
    kmeans = joblib.load("kmeans.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, kmeans, scaler

model, kmeans, scaler = load_models()

# ─────────────────────────────────────────────
# ENCODING MAPS
# ─────────────────────────────────────────────
gender_map    = {"Female": 0, "Male": 1}
marital_map   = {"Divorced": 0, "Married": 1, "Single": 2}
dept_map      = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
role_map      = {
    "Healthcare Representative": 0, "Human Resources": 1, "Laboratory Technician": 2,
    "Manager": 3, "Manufacturing Director": 4, "Research Director": 5,
    "Research Scientist": 6, "Sales Executive": 7, "Sales Representative": 8
}
edu_field_map = {
    "Human Resources": 0, "Life Sciences": 1, "Marketing": 2,
    "Medical": 3, "Other": 4, "Technical Degree": 5
}
travel_map    = {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2}

# ─────────────────────────────────────────────
# LOAD & PREPROCESS DATA  (done BEFORE sidebar so dataset_label is always defined)
# ─────────────────────────────────────────────
# Temporary placeholder — will be replaced after the file-uploader widget renders
uploaded_file = None   # set properly inside sidebar block below

# We need dataset_label defined before the header renders, so we read the
# default dataset first; it will be overwritten if the user uploads a file.
@st.cache_data
def load_default_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

def preprocess(df):
    df = df.copy()
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⬡ Workforce Intelligence</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    st.markdown("**Employee Profile**")

    with st.expander("👤 Personal Info", expanded=True):
        age       = st.slider("Age", 18, 60, 30)
        gender    = st.selectbox("Gender", ["Male", "Female"])
        marital   = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        edu       = st.slider("Education Level (1–5)", 1, 5, 3)
        edu_field = st.selectbox("Education Field",
                                  ["Life Sciences","Medical","Marketing",
                                   "Technical Degree","Human Resources","Other"])

    with st.expander("💼 Job Details"):
        dept      = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
        job_role  = st.selectbox("Job Role",
                                  ["Sales Executive","Research Scientist","Laboratory Technician",
                                   "Manufacturing Director","Healthcare Representative","Manager",
                                   "Sales Representative","Research Director","Human Resources"])
        job_level = st.slider("Job Level (1–5)", 1, 5, 2)
        job_inv   = st.slider("Job Involvement (1–4)", 1, 4, 3)
        job_sat   = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
        env_sat   = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
        rel_sat   = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
        overtime  = st.selectbox("OverTime", ["No", "Yes"])
        travel    = st.selectbox("Business Travel",
                                  ["Non-Travel","Travel_Rarely","Travel_Frequently"])

    with st.expander("💰 Compensation"):
        income       = st.number_input("Monthly Income (₹)", min_value=1000, value=5000, step=500, format="%d")
        daily_rate   = st.number_input("Daily Rate", min_value=100, value=800, step=50, format="%d")
        hourly_rate  = st.number_input("Hourly Rate", min_value=30, value=65, step=5, format="%d")
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, value=14000, step=500, format="%d")
        pct_hike     = st.slider("Salary Hike % Last Year", 11, 25, 14)
        stock_opt    = st.slider("Stock Option Level (0–3)", 0, 3, 1)
        perf_rating  = st.selectbox("Performance Rating", [3, 4],
                                     format_func=lambda x: f"{x} — {'Excellent' if x==3 else 'Outstanding'}")

    with st.expander("📅 Experience"):
        distance     = st.slider("Distance From Home (km)", 1, 30, 9)
        worklife     = st.slider("Work Life Balance (1–4)", 1, 4, 3)
        total_years  = st.slider("Total Working Years", 0, 40, 8)
        years_co     = st.slider("Years At Company", 0, 40, 5)
        years_role   = st.slider("Years In Current Role", 0, 18, 3)
        years_promo  = st.slider("Years Since Last Promotion", 0, 15, 2)
        years_mgr    = st.slider("Years With Current Manager", 0, 17, 3)
        num_cos      = st.slider("Num Companies Worked", 0, 9, 2)
        training     = st.slider("Training Times Last Year", 0, 6, 2)

    predict_clicked = st.button("Run Attrition Analysis →")

# ─────────────────────────────────────────────
# RESOLVE DATASET  (after sidebar so uploaded_file is available)
# ─────────────────────────────────────────────
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    dataset_label = "Custom Dataset"
else:
    raw_df = load_default_data()
    dataset_label = "IBM HR Default Dataset"

dataset_rows = len(raw_df)
df = preprocess(raw_df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <span class="eyebrow">AI · HR Analytics · v2.0</span>
    <h1>Workforce Emotion &amp;<br>Attrition Intelligence</h1>
    <p class="sub">Predict, segment, and understand employee attrition with machine learning</p>
</div>
""", unsafe_allow_html=True)

# FIX: use str() to avoid "undefined" — ensures Python values render correctly
st.markdown(
    f'<div class="dataset-badge">'
    f'<span class="dot"></span>'
    f'{str(dataset_label)} &nbsp;·&nbsp; {int(dataset_rows):,} employees &nbsp;·&nbsp; 16.1% historical attrition rate'
    f'</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_clicked:
    input_data = pd.DataFrame([{
        'Age':                      age,
        'BusinessTravel':           travel_map[travel],
        'DailyRate':                daily_rate,
        'Department':               dept_map[dept],
        'DistanceFromHome':         distance,
        'Education':                edu,
        'EducationField':           edu_field_map[edu_field],
        'EnvironmentSatisfaction':  env_sat,
        'Gender':                   gender_map[gender],
        'HourlyRate':               hourly_rate,
        'JobInvolvement':           job_inv,
        'JobLevel':                 job_level,
        'JobRole':                  role_map[job_role],
        'JobSatisfaction':          job_sat,
        'MaritalStatus':            marital_map[marital],
        'MonthlyIncome':            income,
        'MonthlyRate':              monthly_rate,
        'NumCompaniesWorked':       num_cos,
        'OverTime':                 1 if overtime == "Yes" else 0,
        'PercentSalaryHike':        pct_hike,
        'PerformanceRating':        perf_rating,
        'RelationshipSatisfaction': rel_sat,
        'StockOptionLevel':         stock_opt,
        'TotalWorkingYears':        total_years,
        'TrainingTimesLastYear':    training,
        'WorkLifeBalance':          worklife,
        'YearsAtCompany':           years_co,
        'YearsInCurrentRole':       years_role,
        'YearsSinceLastPromotion':  years_promo,
        'YearsWithCurrManager':     years_mgr,
    }])

    input_data   = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]
    cluster      = int(kmeans.predict(input_scaled)[0])

    if prob > 0.35:
        risk_class, risk_label, risk_desc, risk_icon = (
            "risk-high", "HIGH RISK", "Immediate retention action recommended", "🔴")
    elif prob > 0.20:
        risk_class, risk_label, risk_desc, risk_icon = (
            "risk-medium", "MEDIUM RISK", "Monitor closely — elevated attrition signal", "🟡")
    else:
        risk_class, risk_label, risk_desc, risk_icon = (
            "risk-low", "LOW RISK", "Employee profile appears stable", "🟢")

    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <span class="risk-label">{risk_icon} &nbsp;{risk_label}</span>
            <span class="risk-value">{prob:.0%}</span>
            <span class="risk-desc">{risk_desc}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Attrition Prob</span>
            <span class="metric-value">{prob:.2f}</span>
            <span class="metric-sub">Raw score (0–1)</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-label">Employee Cluster</span>
            <span class="metric-value">{cluster}</span>
            <span class="metric-sub">Segment group (0–3)</span>
        </div>
        """, unsafe_allow_html=True)

    gauge_color = '#ef4444' if prob > 0.35 else '#f59e0b' if prob > 0.20 else '#10b981'
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 34, 'color': '#c7d2f0', 'family': 'Space Grotesk'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#3d4f7c', 'tickfont': {'color': '#3d4f7c'}},
            'bar': {'color': gauge_color},
            'bgcolor': '#1a1a2e', 'bordercolor': '#1e1e3a',
            'steps': [
                {'range': [0,  20], 'color': '#051a10'},
                {'range': [20, 35], 'color': '#1a150a'},
                {'range': [35, 100],'color': '#1f0a0a'},
            ],
            'threshold': {'line': {'color': '#4f6ef7', 'width': 2}, 'thickness': 0.75, 'value': prob * 100}
        },
        title={'text': "Attrition Probability", 'font': {'size': 12, 'color': '#6b7db3', 'family': 'Space Grotesk'}}
    ))
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=230)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("---")

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Attrition Overview</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

if 'Attrition' in df.columns:
    with col1:
        counts = df['Attrition'].value_counts().reset_index()
        counts.columns = ['Attrition', 'Count']
        counts['Label'] = counts['Attrition'].map({0: 'Retained', 1: 'Left'})
        fig1 = px.pie(counts, values='Count', names='Label', title="Attrition Distribution",
                      color_discrete_sequence=['#4f6ef7','#f87171'], hole=0.55)
        fig1.update_traces(textfont=dict(family='Space Grotesk', color='#c7d2f0'))
        fig1.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x='Attrition', y='MonthlyIncome', title="Salary vs Attrition",
                      color='Attrition', color_discrete_map={0:'#4f6ef7', 1:'#f87171'},
                      labels={'Attrition':'','MonthlyIncome':'Monthly Income (₹)'})
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="section-header">Workforce Insights</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

if 'JobRole' in df.columns and 'Attrition' in df.columns:
    with col3:
        fig3 = px.histogram(df, x='JobRole', color='Attrition', title="Attrition by Job Role",
                            barmode='group', color_discrete_map={0:'#4f6ef7', 1:'#f87171'},
                            labels={'JobRole':'','count':'Employees','Attrition':''})
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.scatter(df, x='Age', y='MonthlyIncome',
                      color='Attrition' if 'Attrition' in df.columns else None,
                      title="Age vs Monthly Income", opacity=0.55,
                      color_discrete_map={0:'#4f6ef7', 1:'#f87171'},
                      labels={'MonthlyIncome':'Monthly Income (₹)','Attrition':''})
    fig4.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig4, use_container_width=True)

if 'OverTime' in df.columns and 'Attrition' in df.columns:
    col5, col6 = st.columns(2)
    with col5:
        fig_ot = px.histogram(df, x='OverTime', color='Attrition', title="Overtime vs Attrition",
                              barmode='group', color_discrete_map={0:'#4f6ef7', 1:'#f87171'},
                              labels={'OverTime':'OverTime (0=No, 1=Yes)','Attrition':''})
        fig_ot.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_ot, use_container_width=True)
    with col6:
        fig_dist = px.histogram(df, x='DistanceFromHome', color='Attrition', nbins=20,
                                title="Distance From Home vs Attrition",
                                color_discrete_map={0:'#4f6ef7', 1:'#f87171'},
                                labels={'DistanceFromHome':'Distance (km)','Attrition':''})
        fig_dist.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_dist, use_container_width=True)

st.markdown('<div class="section-header">Feature Correlation</div>', unsafe_allow_html=True)
corr = df.select_dtypes(include=['number']).corr()
fig5 = px.imshow(corr, title="Correlation Heatmap", aspect="auto",
                 color_continuous_scale=[[0,"#1a0a2e"],[0.5,"#1e1e3a"],[1,"#4f6ef7"]])
fig5.update_layout(**PLOTLY_LAYOUT, height=500)
st.plotly_chart(fig5, use_container_width=True)

st.markdown('<div class="section-header">Employee Segmentation</div>', unsafe_allow_html=True)
df_for_scale = df.drop('Attrition', axis=1, errors='ignore')
df_for_scale = df_for_scale.reindex(columns=scaler.feature_names_in_, fill_value=0)
df['Cluster'] = kmeans.predict(scaler.transform(df_for_scale)).astype(str)

fig6 = px.scatter(df, x='MonthlyIncome', y='Age', color='Cluster',
                  title="Employee Clusters (Income vs Age)", opacity=0.65,
                  labels={'MonthlyIncome':'Monthly Income (₹)'},
                  color_discrete_sequence=['#4f6ef7','#7b5cf0','#34d399','#f59e0b'])
fig6.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig6, use_container_width=True)

st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
try:
    rf_model = model.named_estimators_['rf']
    feat_imp = pd.Series(rf_model.feature_importances_, index=scaler.feature_names_in_)
    top_feat = feat_imp.sort_values(ascending=True).tail(12)
    fig7 = px.bar(top_feat, orientation='h', title="Top 12 Predictive Features",
                  labels={'value':'Importance Score','index':''},
                  color=top_feat.values,
                  color_continuous_scale=[[0,'#1e1e3a'],[1,'#4f6ef7']])
    fig7.update_coloraxes(showscale=False)
    fig7.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig7, use_container_width=True)
except Exception:
    st.caption("Feature importance not available for this model type.")

st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.7rem;color:#2a3050;font-family:'JetBrains Mono',monospace;padding:1rem 0;">
    AI-Based Workforce Emotion &amp; Attrition Intelligence &nbsp;·&nbsp; Streamlit + Scikit-learn
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.7rem;color:#2a3050;font-family:'JetBrains Mono',monospace;padding:1rem 0;">
    © 2026 Samridhi Pullani · Workforce Intelligence Dashboard
</div>
""", unsafe_allow_html=True)