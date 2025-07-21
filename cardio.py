import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ------------------ Session State ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "risk_checked" not in st.session_state:
    st.session_state.risk_checked = False
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Login"

# ------------------ Load Users ------------------
if os.path.exists("users.csv"):
    users_df = pd.read_csv("users.csv")
else:
    users_df = pd.DataFrame(columns=["email", "password"])

# ------------------ Load and Prepare Data ------------------
def load_data():
    df = pd.read_csv("cardio_train.csv", sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df["age"] = df["age"] // 365
    return df

cardio_data = load_data()


X = cardio_data.drop("cardio", axis=1)
y = cardio_data["cardio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)


with st.sidebar:
    st.title("CardioCare")

    if st.session_state.logged_in and st.session_state.selected_page == "Login":
        st.session_state.selected_page = "Heart Risk"

    menu_choice = option_menu(
        "Navigation",
        ["Login", "About Us", "Dataset", "Heart Risk"],
        icons=["lock", "info-circle", "table", "heart"],
        default_index=["Login", "About Us", "Dataset", "Heart Risk"].index(st.session_state.selected_page),
        orientation="vertical"
    )

    st.session_state.selected_page = menu_choice




if menu_choice == "Login":
    st.title("User Login")

    if not st.session_state.logged_in:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_row = users_df[users_df["email"] == email]
            if not user_row.empty and user_row.iloc[0]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.selected_page = "Heart Risk"
                st.success("Login successful! Redirecting to Heart Risk...")

            else:
                st.error("Invalid email or password.")
    else:
        st.success(f"Logged in as {st.session_state.user_email}")
        if st.button("Go to Heart Risk Check"):
            st.session_state.selected_page = "Heart Risk"
            st.rerun()

elif menu_choice == "Dataset":
    st.title("Heart Disease Dataset")
    st.subheader("First 10 Records")
    st.dataframe(cardio_data.head(10))

    st.write(f"**Total Records:** {cardio_data.shape[0]}")
    st.write(f"**Total Columns:** {cardio_data.shape[1]}")

    st.subheader("Summary Statistics")
    st.write(cardio_data.describe())

    st.subheader("Heart Disease Risk by Age")
    age_risk = cardio_data[cardio_data["cardio"] == 1]["age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(age_risk.index, age_risk.values, color="salmon", edgecolor="black")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Heart Disease Cases")
    ax.set_title("Heart Disease Frequency by Age")
    st.pyplot(fig)


elif menu_choice == "Heart Risk":
    st.title("Heart Risk Assessment")

    if not st.session_state.logged_in:
        st.warning("Please login first from the sidebar.")
    else:
        st.subheader("Enter your health details:")

        age = st.slider("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 150, 65)
        ap_hi = st.number_input("Systolic BP", 90, 200, 120)
        ap_lo = st.number_input("Diastolic BP", 50, 150, 80)
        cholesterol = st.selectbox("Cholesterol", ["Normal", "Slightly High", "Very High"])
        gluc = st.selectbox("Glucose", ["Normal", "Slightly High", "Very High"])
        smoke = st.radio("Do you smoke?", ["No", "Yes"])
        alco = st.radio("Do you drink alcohol?", ["No", "Yes"])
        active = st.radio("Are you active?", ["Yes", "No"])

        if st.button("Check Risk"):
            gender_val = 1 if gender == "Male" else 2
            chol_val = {"Normal": 1, "Slightly High": 2, "Very High": 3}[cholesterol]
            gluc_val = {"Normal": 1, "Slightly High": 2, "Very High": 3}[gluc]
            smoke_val = 1 if smoke == "Yes" else 0
            alco_val = 1 if alco == "Yes" else 0
            active_val = 1 if active == "Yes" else 0
            age_days = age * 365

            input_data = np.array([[age_days, gender_val, height, weight, ap_hi, ap_lo,
                                    chol_val, gluc_val, smoke_val, alco_val, active_val]])
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            bmi = round(weight / ((height / 100) ** 2), 2)
            bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

            st.subheader("Results")
            st.write(f"**Your BMI:** {bmi} ({bmi_status})")
            st.write(f"**Model accuracy on test data:** {accuracy:.2%}")

            if prediction == 1:
                st.error("Heart Disease Risk: High")
                st.markdown("**Recommendation**")
                st.warning("Please consult a cardiologist as soon as possible.")
                st.markdown("### Health Tips for High Risk")
                st.markdown("""
                - Avoid oily, fried, and salty foods  
                - Walk or exercise at least 30 minutes daily  
                - Quit smoking and reduce alcohol  
                - Monitor your blood pressure regularly  
                - Take medications as prescribed by your doctor  
                - Practice stress-reducing activities like yoga or meditation  
                """)
            else:
                st.success("Heart Disease Risk: Low")
                st.markdown("### Health Tips to Stay Healthy")
                st.markdown("""
                - Maintain a balanced diet with fruits and vegetables  
                - Exercise regularly and stay active  
                - Avoid too much sugar or fast food  
                - Sleep 7–8 hours every day  
                - Get regular health checkups  
                - Keep stress levels low  
                """)

            st.subheader("Risk Probability")
            fig, ax = plt.subplots()
            ax.pie(proba, labels=['Healthy', 'Heart Disease'], autopct='%1.1f%%',
                   startangle=90, explode=(0, 0.1), shadow=True, colors=['lightgreen', 'salmon'])
            ax.axis('equal')
            st.pyplot(fig)

            st.session_state.risk_checked = True

elif menu_choice == "About Us":
    st.title("About CardioCare")
    st.markdown("""
    **CardioCare** helps you:
    - Predict heart disease risk using AI  
    - Track your BMI and blood pressure  
    - View dataset and trends  

    > This app is for educational use and doesn’t replace medical advice.
    """)
