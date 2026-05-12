"""
Spinal Surgery Prediction Tool
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spinal Surgery Risk Predictor",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Authentication ─────────────────────────────────────────────────────────────
VALID_CREDENTIALS = {"Jay": "Rockford"}

def login_screen():
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none}
        .login-title {font-size:2rem; font-weight:700; color:#1F4E79; text-align:center; margin-bottom:4px}
        .login-sub   {font-size:0.95rem; color:#666; text-align:center; margin-bottom:24px}
    </style>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1.2, 1, 1.2])
    with col:
        st.markdown('<p class="login-title">🦴 Spinal Surgery</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-sub">Risk Predictor — Please sign in</p>', unsafe_allow_html=True)
        st.markdown("---")
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("🔐 Sign In", type="primary", use_container_width=True):
            if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

def check_auth():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        login_screen()
        st.stop()

check_auth()

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2rem; font-weight:700; color:#1F4E79; margin-bottom:0}
    .sub-header  {font-size:1rem; color:#555; margin-bottom:1.5rem}
    .pred-box    {border-radius:12px; padding:20px; text-align:center; margin:8px 0}
    .pred-short  {background:#D4EDDA; border:2px solid #28A745}
    .pred-medium {background:#FFF3CD; border:2px solid #FFC107}
    .pred-long   {background:#F8D7DA; border:2px solid #DC3545}
    .pred-low    {background:#D4EDDA; border:2px solid #28A745}
    .pred-high   {background:#F8D7DA; border:2px solid #DC3545}
    .label-big   {font-size:1.6rem; font-weight:700}
    .label-sub   {font-size:0.9rem; color:#444; margin-top:4px}
    .section     {font-size:1.1rem; font-weight:600; color:#1F4E79;
                  border-bottom:2px solid #1F4E79; padding-bottom:4px; margin:1rem 0 0.5rem}
    .disclaimer  {background:#F0F4F8; border-left:4px solid #1F4E79;
                  padding:10px 16px; border-radius:6px; font-size:0.82rem; color:#555}
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    los_model = joblib.load('los_model.pkl')
    rd_model  = joblib.load('rd_model.pkl')
    le_los    = joblib.load('los_label_encoder.pkl')
    feat_cols = joblib.load('feature_cols.pkl')
    return los_model, rd_model, le_los, feat_cols

los_model, rd_model, le_los, FEATURE_COLS = load_models()

# ── Header + Sign Out ─────────────────────────────────────────────────────────
hdr_col, logout_col = st.columns([5, 1])
with hdr_col:
    st.markdown('<p class="main-header">🦴 Spinal Surgery Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pre-operative prediction of Hospital LOS and 30-Day Readmission Risk</p>', unsafe_allow_html=True)
with logout_col:
    st.markdown(f"<br><small>👤 {st.session_state['username']}</small>", unsafe_allow_html=True)
    if st.button("Sign Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

st.markdown('<div class="disclaimer">⚠️ <b>Clinical Disclaimer:</b> This tool is for decision support only. Predictions are based on historical registry data and should not replace clinical judgment.</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar — Input Form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Information")
    st.markdown("Fill in all known fields. Unknown values can be left blank.")

    st.markdown('<p class="section">Demographics</p>', unsafe_allow_html=True)
    gender         = st.selectbox("Gender", ["Female (0)", "Male (1)"])
    age            = st.number_input("Age (years)", min_value=1, max_value=110, value=55)
    bmi            = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.0, step=0.1)
    ethnicity      = st.selectbox("Ethnicity", ["Unknown", "White (1)", "African American (2)", "Hispanic (3)", "Asian (4)", "Other (5)"])
    insurance      = st.selectbox("Insurance", ["Unknown", "Medicare/Medicaid (1)", "Worker's Comp (2)", "Private/Other (3)", "Self-pay (4)"])

    st.markdown('<p class="section">Surgical Plan</p>', unsafe_allow_html=True)
    spine_region   = st.selectbox("Spine Region", ["Unknown", "Cervical (1)", "Thoracic (2)", "Lumbar (3)", "Thoracolumbar (4)", "Cervicothoracic (5)"])
    procedure_type = st.selectbox("Procedure Type", ["Unknown", "MIS (1)", "Mini-Open (2)", "Microscopic (3)", "Regular Open (4)", "UBE (5)", "Full Endo (6)"])
    approach       = st.selectbox("Approach", ["Unknown", "Anterior (1)", "Posterior (2)", "TLIF (3)", "Lateral (4)", "Combined AP (5)"])
    fusion1_type   = st.selectbox("Fusion Type", ["None (0)", "ACDF (1)", "TLIF (2)", "ALIF (3)", "LLIF/XLIF (4)", "PCF (5)", "ATF (6)", "PTF (7)", "PLF (8)", "Other (9)", "CDR (10)", "LDR (11)"])
    fusion1_levels = st.number_input("Fusion Levels (Fusion 1)", min_value=0, max_value=12, value=1)
    fusion2_levels = st.number_input("Fusion Levels (Fusion 2, if applicable)", min_value=0, max_value=12, value=0)
    laminectomy    = st.selectbox("Laminectomy", ["No (0)", "Yes (1)"])
    lam_levels     = st.number_input("Laminectomy Levels", min_value=0, max_value=12, value=0)
    decomp         = st.selectbox("Decompression", ["None (0)", "Foraminotomy (1)", "Facetectomy (2)", "Laminotomy (3)", "Laminoplasty (4)", "Laminoforaminotomy (5)", "Annulotomy (6)", "Other (7)"])
    decomp_levels  = st.number_input("Decompression Levels", min_value=0, max_value=12, value=0)
    discectomy     = st.selectbox("Discectomy", ["None (0)", "Discectomy (1)", "Far Lateral (2)"])
    disc_levels    = st.number_input("Discectomy Levels", min_value=0, max_value=12, value=0)
    ebl            = st.number_input("Estimated Blood Loss (mL)", min_value=0, max_value=5000, value=100)
    or_time        = st.number_input("OR Time (minutes)", min_value=0, max_value=900, value=90)
    asa            = st.selectbox("ASA Score", ["Unknown", "1", "2", "3", "4", "5"])
    procedure_flag = st.selectbox("Procedure Status", ["Unknown", "Primary (1)", "Primary Fusion + Revision Decomp (2)", "Revision (3)", "Reoperation (4)"])

    st.markdown('<p class="section">Diagnosis</p>', unsafe_allow_html=True)
    hnp           = st.selectbox("HNP", ["No (0)", "Yes (1)", "Massive HNP (2)", "Far Lateral HNP (3)"])
    stenosis      = st.selectbox("Central/Spinal Stenosis", ["No (0)", "Yes (1)"])
    foraminal     = st.selectbox("Foraminal Stenosis", ["No (0)", "Yes (1)"])
    spondy        = st.selectbox("Spondylolisthesis", ["No (0)", "Yes (1)"])
    ddd           = st.selectbox("DDD", ["No (0)", "Yes (1)"])
    neuropathy    = st.selectbox("Neuropathy", ["None (0)", "Radiculopathy (1)", "Myelopathy (2)", "Myeloradiculopathy (3)"])
    neuro_deficit = st.selectbox("Neurological Deficit", ["No (0)", "Yes (1)"])
    motor_deficit = st.selectbox("Motor Deficit", ["None (0)", "Palsy (1)", "Foot Drop (2)", "Bowel/Bladder (3)", "Atrophy (4)", "Cauda Equina (5)", "Weakness (6)", "Paraplegia (7)"])
    sensory       = st.selectbox("Sensory Deficit", ["No (0)", "Yes (1)"])
    deformity     = st.selectbox("Deformity", ["None (0)", "Scoliosis (1)", "Kyphosis (2)", "Flatback (3)", "Lordosis (4)"])
    fracture      = st.selectbox("Fracture", ["None (0)", "Traumatic (1)", "Pathologic (2)", "Compression (3)", "Burst (4)", "Pars (5)", "Endplate (6)"])
    traumatic     = st.selectbox("Traumatic Mechanism", ["No (0)", "Yes (1)"])

    st.markdown('<p class="section">Comorbidities</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        smoker     = st.checkbox("Smoker")
        diabetes_u = st.checkbox("Diabetes (uncomplicated)")
        diabetes_c = st.checkbox("Diabetes (complicated)")
        htn        = st.checkbox("Hypertension")
        mi         = st.checkbox("MI")
        chf        = st.checkbox("CHF")
        pvd        = st.checkbox("Peripheral Vascular Dz")
        sleep_ap   = st.checkbox("Sleep Apnea")
    with c2:
        neuro_dz   = st.checkbox("Neurologic Disease")
        arthritis  = st.checkbox("Arthritis")
        cancer     = st.checkbox("Cancer")
        metastasis = st.checkbox("Metastasis")
        liver_dz   = st.checkbox("Liver Disease")
        renal_fail = st.checkbox("Renal Failure")
        lung_dz    = st.checkbox("Chronic Lung Disease")
        gi_bleed   = st.checkbox("GI Bleed")
    cci = st.number_input("CCI Score (with age)", min_value=0, max_value=30, value=0)

    st.markdown('<p class="section">History</p>', unsafe_allow_html=True)
    prior_cerv  = st.selectbox("Prior Cervical Surgery", ["No (0)", "Yes (1)"])
    prior_tl    = st.selectbox("Prior Thoracolumbar Surgery", ["No (0)", "Yes (1)"])
    symptom_dur = st.number_input("Symptom Duration (months, 0=unknown)", min_value=0, max_value=360, value=0)

    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_sel(s):
    try:
        return int(s.split("(")[-1].replace(")", "").strip())
    except:
        return np.nan

def parse_sel_unknown(s):
    if "Unknown" in s:
        return np.nan
    return parse_sel(s)

def build_input():
    f1_lev = float(fusion1_levels); f2_lev = float(fusion2_levels)
    lam_lev = float(lam_levels);    dec_lev = float(decomp_levels)
    disc_lev = float(disc_levels)
    comorbidity_count = sum([smoker, mi, diabetes_u, diabetes_c, htn, neuro_dz,
                             arthritis, chf, pvd, cancer, metastasis, liver_dz,
                             renal_fail, lung_dz, gi_bleed, sleep_ap])
    proc_flag_val = parse_sel_unknown(procedure_flag)
    spine_val     = parse_sel_unknown(spine_region)
    proc_type_val = parse_sel_unknown(procedure_type)
    row = {
        'Gender_Male': parse_sel(gender), 'BMI': bmi, 'Age': float(age),
        'Ethnicity': parse_sel_unknown(ethnicity), 'Insurance': parse_sel_unknown(insurance),
        'Spine_Region': spine_val, 'Spondylolisthesis': parse_sel(spondy),
        'Isthmic_Spondy_Grade': np.nan, 'Degen_Spondy_Grade': np.nan,
        'HNP': parse_sel(hnp), 'DDD': parse_sel(ddd),
        'Central_Stenosis': parse_sel(stenosis), 'Foraminal_Stenosis': parse_sel(foraminal),
        'Fracture': parse_sel(fracture), 'Neuropathy': parse_sel(neuropathy),
        'Neurological_Deficit': parse_sel(neuro_deficit), 'Motor_Deficit': parse_sel(motor_deficit),
        'Sensory_Deficit': parse_sel(sensory), 'Deformity': parse_sel(deformity),
        'Infection': 0.0, 'Other_Dx': 0.0, 'Traumatic': parse_sel(traumatic),
        'Smoker': float(smoker), 'MI': float(mi),
        'Diabetes_Uncomplicated': float(diabetes_u), 'Diabetes_Complicated': float(diabetes_c),
        'Hypertension': float(htn), 'Neurologic_Disease': float(neuro_dz),
        'Arthritis': float(arthritis), 'CHF': float(chf), 'PVD': float(pvd),
        'Cancer': float(cancer), 'Metastasis': float(metastasis),
        'Liver_Disease': float(liver_dz), 'Renal_Failure': float(renal_fail),
        'Chronic_Lung_Disease': float(lung_dz), 'GI_Bleed': float(gi_bleed),
        'Sleep_Apnea': float(sleep_ap), 'CCI_with_age': float(cci),
        'ASA_Score': parse_sel_unknown(asa), 'Surgery_Location': np.nan,
        'Procedure_Type': proc_type_val, 'Approach': parse_sel_unknown(approach),
        'Fusion1_Type': parse_sel(fusion1_type), 'Fusion1_NumLevel': f1_lev,
        'Fusion2_Type': np.nan, 'Fusion2_NumLevel': f2_lev,
        'Laminectomy': parse_sel(laminectomy), 'Laminectomy_NumLevel': lam_lev,
        'Decompression': parse_sel(decomp), 'Decompression_NumLevel': dec_lev,
        'Discectomy': parse_sel(discectomy), 'Discectomy_NumLevel': disc_lev,
        'Corpectomy': 0.0, 'EBL_mL': float(ebl), 'OR_Time_min': float(or_time),
        'Transfusion': 0.0, 'IntraOp_Complication': 0.0, 'Durotomy': 0.0,
        'Pedicle_Screw': np.nan, 'Cage_Instrumentation': np.nan, 'BMP2_Use': 0.0,
        'Prior_Cervical_Surgery': parse_sel(prior_cerv), 'Prior_TL_Surgery': parse_sel(prior_tl),
        'Prior_Conservative_Tx': np.nan, 'Procedure_Primary_Flag': proc_flag_val,
        'Symptom_Duration': float(symptom_dur) if symptom_dur > 0 else np.nan,
        'Total_Fusion_Levels': f1_lev + f2_lev,
        'Total_Surgical_Levels': f1_lev + f2_lev + lam_lev + dec_lev + disc_lev,
        'Comorbidity_Count': float(comorbidity_count),
        'Is_Revision': float(proc_flag_val >= 3) if not np.isnan(proc_flag_val) else 0.0,
        'Is_Combined_Approach': float(parse_sel_unknown(approach) == 5),
        'Is_Open_Surgery': float(proc_type_val == 4) if not np.isnan(proc_type_val) else 0.0,
        'Any_Fusion': float(parse_sel(fusion1_type) > 0),
        'Has_Neurological_Issue': float(parse_sel(neuro_deficit) > 0 or parse_sel(motor_deficit) > 0),
        'Is_Cervical': float(spine_val == 1) if not np.isnan(spine_val) else 0.0,
        'Is_Lumbar': float(spine_val == 3) if not np.isnan(spine_val) else 0.0,
    }
    return pd.DataFrame([row])[FEATURE_COLS]

def shap_chart(model, X_input, feature_cols, title):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        sv = shap_values[int(model.predict(X_input)[0])][0]
    else:
        sv = shap_values[0]
    feat_df = pd.DataFrame({'Feature': feature_cols, 'SHAP': sv})
    feat_df = feat_df.reindex(feat_df['SHAP'].abs().sort_values(ascending=False).index).head(12).sort_values('SHAP')
    colors  = ['#DC3545' if v > 0 else '#28A745' for v in feat_df['SHAP']]
    fig = go.Figure(go.Bar(
        x=feat_df['SHAP'], y=feat_df['Feature'], orientation='h',
        marker_color=colors, text=[f"{v:+.3f}" for v in feat_df['SHAP']], textposition='outside'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color='#1F4E79')),
        xaxis_title="SHAP Value (red = increases risk, green = reduces risk)",
        height=420, margin=dict(l=20, r=60, t=50, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#eee'), yaxis=dict(gridcolor='#eee')
    )
    return fig

# ── Main panel ────────────────────────────────────────────────────────────────
if not predict_btn:
    st.info("👈  Fill in the patient details in the sidebar and click **Predict** to generate predictions.")
    c1, c2, c3 = st.columns(3)
    c1.metric("LOS Classes",    "Short ≤1d / Medium 2-3d / Long ≥4d")
    c2.metric("Readmission",    "Binary: Any 30-day return")
    c3.metric("Explainability", "SHAP feature attribution")
else:
    X_input        = build_input()
    los_pred_enc   = los_model.predict(X_input)[0]
    los_probas     = los_model.predict_proba(X_input)[0]
    los_pred_label = le_los.inverse_transform([los_pred_enc])[0]
    los_classes    = le_los.classes_
    rd_proba       = rd_model.predict_proba(X_input)[0][1]
    rd_risk        = "HIGH" if rd_proba >= 0.15 else ("MODERATE" if rd_proba >= 0.05 else "LOW")

    st.markdown("## 🔍 Prediction Results")
    col_los, col_rd = st.columns(2)

    with col_los:
        st.markdown("### 🏥 Hospital Length of Stay")
        css    = {'Short_0_1d':'pred-short','Medium_2_3d':'pred-medium','Long_4plus_d':'pred-long'}
        labels = {'Short_0_1d':'Short  ≤ 1 day','Medium_2_3d':'Medium  2–3 days','Long_4plus_d':'Long  ≥ 4 days'}
        box_cls = css.get(los_pred_label, 'pred-medium')
        box_lbl = labels.get(los_pred_label, los_pred_label)
        st.markdown(f'<div class="pred-box {box_cls}"><div class="label-big">{box_lbl}</div><div class="label-sub">Predicted LOS Category</div></div>', unsafe_allow_html=True)
        st.markdown("**Class probabilities**")
        st.dataframe(pd.DataFrame({'Category': [labels.get(c,c) for c in los_classes],
                                   'Probability': [f"{p:.1%}" for p in los_probas]}),
                     hide_index=True, use_container_width=True)

    with col_rd:
        st.markdown("### 🔄 30-Day Readmission Risk")
        rd_css = {'HIGH':'pred-high','MODERATE':'pred-medium','LOW':'pred-low'}[rd_risk]
        st.markdown(f'<div class="pred-box {rd_css}"><div class="label-big">{rd_risk}</div><div class="label-sub">Predicted Risk Level &nbsp;|&nbsp; Probability: {rd_proba:.1%}</div></div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(rd_proba*100,1), number={'suffix':'%'},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'#DC3545' if rd_risk=='HIGH' else ('#FFC107' if rd_risk=='MODERATE' else '#28A745')},
                   'steps':[{'range':[0,5],'color':'#D4EDDA'},{'range':[5,15],'color':'#FFF3CD'},{'range':[15,100],'color':'#F8D7DA'}],
                   'threshold':{'line':{'color':'black','width':3},'value':rd_proba*100}},
            title={'text':"30-Day Return Probability"}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("## 🧠 Why This Prediction? (SHAP Explanation)")
    st.caption("Red bars increase predicted risk/LOS. Green bars reduce it. Bar length = strength of influence.")
    s1, s2 = st.columns(2)
    with s1:
        with st.spinner("Computing LOS explanation..."):
            st.plotly_chart(shap_chart(los_model, X_input, FEATURE_COLS, f"LOS Drivers — Predicted: {box_lbl}"), use_container_width=True)
    with s2:
        with st.spinner("Computing readmission explanation..."):
            st.plotly_chart(shap_chart(rd_model, X_input, FEATURE_COLS, f"Readmission Drivers — Risk: {rd_risk} ({rd_proba:.1%})"), use_container_width=True)

    with st.expander("📄 View patient input data used for prediction"):
        display_df = X_input.T.rename(columns={0:'Value'})
        st.dataframe(display_df[display_df['Value'].notna()], use_container_width=True)
