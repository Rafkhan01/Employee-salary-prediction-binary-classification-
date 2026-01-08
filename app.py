import streamlit as st
import pandas as pd
import pickle

# Loading
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: model.pkl or encoders.pkl not found. Please run the notebook export cell first.")

st.set_page_config(page_title="Income Predictor", layout="centered")
st.title("💰 Adult Income Predictor")
st.write("Enter profile details to predict if income exceeds $50K/year.")

# SIDEBAR
st.sidebar.header("User Profile")

def get_user_input():
    age = st.sidebar.slider("Age", 17, 90, 30)
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=200000)
    edu_num = st.sidebar.slider("Education Num (Years)", 1, 16, 9)
    cap_gain = st.sidebar.number_input("Capital Gain", value=0)
    cap_loss = st.sidebar.number_input("Capital Loss", value=0)
    hours = st.sidebar.slider("Hours per Week", 1, 99, 40)
    
    # Categorical Inputs (using saved encoder classes)
    workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
    marital = st.sidebar.selectbox("Marital Status", encoders['marital.status'].classes_)
    occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.sidebar.selectbox("Race", encoders['race'].classes_)
    sex = st.sidebar.selectbox("Sex", encoders['sex'].classes_)
    country = st.sidebar.selectbox("Native Country", encoders['native.country'].classes_)

    # Prepare data for prediction
    data = {
        'age': age,
        'workclass': encoders['workclass'].transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education.num': edu_num,
        'marital.status': encoders['marital.status'].transform([marital])[0],
        'occupation': encoders['occupation'].transform([occupation])[0],
        'relationship': encoders['relationship'].transform([relationship])[0],
        'race': encoders['race'].transform([race])[0],
        'sex': encoders['sex'].transform([sex])[0],
        'capital.gain': cap_gain,
        'capital.loss': cap_loss,
        'hours.per.week': hours,
        'native.country': encoders['native.country'].transform([country])[0]
    }
    return pd.DataFrame(data, index=[0])

df = get_user_input()

#PREDICTION
st.subheader("Prediction Result")
prediction_proba = model.predict_proba(df)[0][1]
prediction = model.predict(df)[0]

# Handling if your model returns ' >50K' or 1
is_high_income = str(prediction).strip() == '>50K' or prediction == 1

if is_high_income:
    st.success(f"Prediction: **>50K Income** (Probability: {round(prediction_proba*100, 2)}%)")
else:
    st.warning(f"Prediction: **<=50K Income** (Probability: {round((1-prediction_proba)*100, 2)}%)")

#  WHAT-IF ANALYSIS
st.divider()
st.subheader("🧐 What-If Analysis")
st.info("Adjust values below to see how they impact the probability of earning >50K.")

col1, col2 = st.columns(2)
with col1:
    wi_hours = st.slider("Adjust Hours per Week", 1, 99, int(df['hours.per.week'][0]))
with col2:
    wi_gain = st.number_input("Adjust Capital Gain", value=int(df['capital.gain'][0]))

# Updating
df_wi = df.copy()
df_wi['hours.per.week'] = wi_hours
df_wi['capital.gain'] = wi_gain

wi_proba = model.predict_proba(df_wi)[0][1]
diff = wi_proba - prediction_proba

st.write(f"**New Probability of earning >50K:** {round(wi_proba*100, 2)}%")
st.progress(wi_proba)

if diff > 0.01:
    st.success(f"That change increased probability by {round(diff*100, 2)}%!")
elif diff < -0.01:
    st.error(f"That change decreased probability by {round(abs(diff)*100, 2)}%.")
else:

    st.write("Small changes don't significantly impact the current prediction.")
