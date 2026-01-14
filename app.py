import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
@st.cache_resource
def load_model():
    model = joblib.load("models/cardio_cloud_compatible.pkl")
    feature_cols = joblib.load("model_artifacts/feature_columns.pkl")
    return model, feature_cols

model, feature_cols = load_model()

st.title("🫀 Cardiovascular Disease Risk Prediction")
st.markdown("*Enter patient information to get instant risk assessment*")

# Sidebar inputs
st.sidebar.header("👤 Patient Information")

age_years = st.sidebar.slider("Age (years)", 18, 90, 50)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 75)
ap_hi = st.sidebar.slider("Systolic BP (mmHg)", 80, 240, 120)
ap_lo = st.sidebar.slider("Diastolic BP (mmHg)", 40, 200, 80)

# Calculated features
bmi = weight / ((height / 100) ** 2)
high_ap_hi = 1 if ap_hi >= 140 else 0
high_ap_lo = 1 if ap_lo >= 90 else 0

# Categorical inputs
age_group = st.sidebar.selectbox("Age Group", ["<40", "40-60", "60+"])
gender = st.sidebar.selectbox("Gender", ["Female (1)", "Male (2)"])
gender = int(gender.split("(")[1].split(")")[0])
smoke = st.sidebar.selectbox("Smokes?", ["No (0)", "Yes (1)"])
smoke = int(smoke.split("(")[1].split(")")[0])
alco = st.sidebar.selectbox("Alcohol?", ["No (0)", "Yes (1)"])
alco = int(alco.split("(")[1].split(")")[0])
active = st.sidebar.selectbox("Physically active?", ["No (0)", "Yes (1)"])
active = int(active.split("(")[1].split(")")[0])
cholesterol = st.sidebar.selectbox("Cholesterol", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
cholesterol = int(cholesterol.split("(")[1].split(")")[0])
gluc = st.sidebar.selectbox("Glucose", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
gluc = int(gluc.split("(")[1].split(")")[0])

# Predict button
if st.button("🔮 Predict Risk", type="primary"):
    # Create input matching exact training features
    input_data = {
        "age_years": float(age_years),
        "height": float(height),
        "weight": float(weight),
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "bmi": float(bmi),
        "age_group": age_group,
        "smoke": int(smoke),
        "alco": int(alco),
        "active": int(active),
        "high_ap_hi": int(high_ap_hi),
        "high_ap_lo": int(high_ap_lo),
        "cholesterol": int(cholesterol),
        "gluc": int(gluc),
        "gender": int(gender),
    }

    # CRITICAL: Use saved feature_cols for correct column ordering
    input_df = pd.DataFrame([input_data])[feature_cols]

    # Predict
    try:
        risk_proba = model.predict_proba(input_df)[:, 1][0]
        prediction = int(risk_proba >= 0.5)

        # Display result
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns([1, 2])
        with col1:
            if prediction == 1:
                st.error("⚠️ **HIGH RISK**")
            else:
                st.success("✅ **LOW RISK**")

        with col2:
            st.metric("Disease Probability", f"{risk_proba:.1%}")

        # Risk interpretation
        if risk_proba >= 0.7:
            st.warning("🚨 **Very High Risk**: Immediate medical consultation recommended")
        elif risk_proba >= 0.5:
            st.warning("⚠️ **Elevated Risk**: Consider scheduling a checkup")
        elif risk_proba >= 0.3:
            st.info("ℹ️ **Moderate Risk**: Monitor health indicators regularly")
        else:
            st.success("✅ **Low Risk**: Maintain healthy lifestyle")

        # Show input data
        with st.expander("📋 View Input Data"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

st.markdown("---")
st.caption("⚕️ Powered by scikit-learn Random Forest | For educational purposes only - not medical advice")