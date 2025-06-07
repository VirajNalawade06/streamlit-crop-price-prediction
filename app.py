import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')

df = load_data()

# Encode categorical columns
df_encoded = df.copy()
le_state = LabelEncoder()
le_crop = LabelEncoder()
df_encoded['State'] = le_state.fit_transform(df_encoded['State'])
df_encoded['Crop'] = le_crop.fit_transform(df_encoded['Crop'])

features = df_encoded.drop(columns=['Price'])
target = df_encoded['Price']

# UI Header
st.markdown("<h1 style='color:green;'>🌾 Crop Price Prediction App</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🧠 Train Model", "🔮 Predict Price"])

# Tab 1: Data Preview
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_encoded.corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# Tab 2: Train Model
with tab2:
    st.subheader("Choose Model and Train")
    model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression"])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 Mean Absolute Error: **{mae:.2f}**")
    st.write(f"📈 R² Score: **{r2:.2f}**")

    # Feature importance
    if model_choice == "Random Forest":
        st.subheader("📌 Feature Importance")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        st.bar_chart(feat_df.set_index('Feature'))

# Tab 3: Prediction
with tab3:
    st.subheader("Enter Input to Predict Price")

    state_input = st.selectbox("State", df['State'].unique())
    crop_input = st.selectbox("Crop", df['Crop'].unique())
    cost1 = st.number_input("Cost of Cultivation 1", min_value=0.0)
    cost2 = st.number_input("Cost of Cultivation 2", min_value=0.0)
    production = st.number_input("Production", min_value=0.0)
    yield_val = st.number_input("Yield", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([[
            le_state.transform([state_input])[0],
            le_crop.transform([crop_input])[0],
            cost1, cost2, production, yield_val, temperature, rainfall
        ]], columns=features.columns)

        pred_price = model.predict(input_df)[0]
        st.success(f"💰 Predicted Crop Price: ₹{pred_price:,.2f}")
