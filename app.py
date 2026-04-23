import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from streamlit_option_menu import option_menu

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AdmitScope", layout="wide")

# ==============================
# DARK ORANGE UI CSS
# ==============================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: #0d0d0d;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #121212;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #f97316, #fb923c);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

/* Sliders */
.stSlider > div[data-baseweb="slider"] > div > div {
    background-color: #f97316 !important;
}

/* Titles */
h1, h2, h3 {
    color: #f1f5f9;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("Admission_Predict.csv")
    if "Serial No." in df.columns:
        df.drop("Serial No.", axis=1, inplace=True)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    return df

df = load_data()

X = df.drop("Chance_of_Admit", axis=1)
y = df["Chance_of_Admit"]

# ==============================
# MODEL
# ==============================
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# ==============================
# SHAP EXPLAINER
# ==============================
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

# ==============================
# SIDEBAR (ICON MENU 🔥)
# ==============================
with st.sidebar:
    selected = option_menu(
        "Admission Lelo Mitron",
        ["Home", "Predictor", "Dashboard"],
        icons=["house", "cpu", "bar-chart"],
        menu_icon="mortarboard",
        default_index=0,
    )

# ==============================
# HOME PAGE
# ==============================
if selected == "Home":

    st.markdown("<h1 style='text-align:center;'>GradPredict: AI-Based Admission Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Predict your admission chances using Machine Learning & AI</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Start Prediction"):
            st.session_state.page = "Predictor"

    with col2:
        if st.button("📊 View Analytics"):
            st.session_state.page = "Dashboard"

# ==============================
# PREDICTOR PAGE
# ==============================
elif selected == "Predictor":

    st.title("Admit Predictor & What-If Analysis")
    st.caption("Adjust inputs to see real-time changes")

    col1, col2 = st.columns([1,1])

    # INPUTS
    with col1:
        st.markdown("### Profile Parameters")

        gre = st.slider("GRE Score", 260, 340, 310)
        toefl = st.slider("TOEFL Score", 0, 120, 105)
        cgpa = st.slider("CGPA", 6.0, 10.0, 8.5)
        university_rating = st.slider("University Rating", 1, 5, 3)
        sop = st.slider("SOP", 1.0, 5.0, 3.5)
        lor = st.slider("LOR", 1.0, 5.0, 3.5)
        research = st.radio("Research", [0,1])

    # RESULT
    with col2:
        input_df = pd.DataFrame(
            [[gre, toefl, university_rating, sop, lor, cgpa, research]],
            columns=X.columns
        )

        prediction = model.predict(input_df)[0]
        prediction_percent = round(prediction * 100, 1)

        st.markdown("### Admission Chance")

        # 🎯 Circular Ring
        st.markdown(f"""
        <div style="display:flex; justify-content:center; margin-top:20px;">
            <div style="
                width:180px;
                height:180px;
                border-radius:50%;
                background: conic-gradient(
                    #f97316 {prediction_percent}%,
                    #262626 {prediction_percent}%
                );
                display:flex;
                align-items:center;
                justify-content:center;
                box-shadow: 0 0 20px rgba(249,115,22,0.6);
            ">
                <div style="
                    width:130px;
                    height:130px;
                    background:#0d0d0d;
                    border-radius:50%;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:28px;
                    font-weight:bold;
                    color:white;
                ">
                    {prediction_percent}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction > 0.8:
            st.success("🔥 High Chance")
        elif prediction > 0.6:
            st.warning("👍 Moderate Chance")
        else:
            st.error("⚠️ Low Chance")

        # FEATURE IMPACT
        st.markdown("### Feature Impact (Explainable AI)")

        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "Impact": np.abs(shap_values[0])
        }).sort_values(by="Impact")

        fig = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#f97316"],
            title="Feature Importance"
        )

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# DASHBOARD PAGE
# ==============================
elif selected == "Dashboard":

    st.title("Analytics Dashboard")
    st.caption("Overview of applicant data and ML insights")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Applicants", len(df))
    c2.metric("Avg Admit Chance", f"{round(df['Chance_of_Admit'].mean()*100,1)}%")
    c3.metric("Average GRE", round(df["GRE_Score"].mean(),1))
    c4.metric("Research Exp.", f"{round(df['Research'].mean()*100,1)}%")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="GRE_Score",
            nbins=20,
            color_discrete_sequence=["#f97316"],
            title="GRE Distribution"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="GRE_Score",
            y="Chance_of_Admit",
            color="Chance_of_Admit",
            color_continuous_scale="Oranges",
            title="GRE vs Admit Chance"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Oranges",
        title="Correlation Heatmap"
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)