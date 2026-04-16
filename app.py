import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================= PAGE SETUP =================
st.set_page_config(page_title="Churn AI Dashboard", layout="wide")

# ================= PREMIUM UI =================
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #38bdf8;
}
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #2563eb);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))

# ================= LOAD DATA =================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ================= TITLE =================
st.title("📊 Premium Customer Churn Dashboard")

# ================= ACCURACY =================
st.markdown("## 🎯 Model Accuracy")
st.info(f"{round(accuracy * 100, 2)} %")

# ================= INPUT (NO DEFAULT VALUES) =================
st.subheader("Enter Customer Details")

tenure = st.text_input("⏳ Tenure (Months)")
monthly = st.text_input("💰 Monthly Charges")
total = st.text_input("📊 Total Charges")

# ================= PREDICTION =================
if st.button("🔍 Predict"):

    # validation
    if not (tenure and monthly and total):
        st.warning("⚠️ Please enter all values")
        st.stop()

    # convert
    tenure = float(tenure)
    monthly = float(monthly)
    total = float(total)

    prediction = model.predict([[tenure, monthly, total]])

    st.markdown("## 📌 Result")

    if prediction[0] == 1:
        st.error("❌ Customer WILL LEAVE")
        risk = "High"
    else:
        st.success("✅ Customer WILL STAY")
        risk = "Low"

    st.metric("Customer Risk Level", risk)

    # ================= GRAPH 1 =================
    st.markdown("## 📊 Feature Comparison")

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    ax1.bar(
        ["Tenure", "Monthly", "Total"],
        [tenure, monthly, total],
        color=["#38bdf8", "#60a5fa", "#2563eb"]
    )

    ax1.set_xlabel("Features")
    ax1.set_ylabel("Values")
    ax1.set_title("Customer Data Overview")
    ax1.grid(True, alpha=0.2)

    st.pyplot(fig1)

    # ================= GRAPH 2 =================
    st.markdown("## 📈 Churn Risk")

    labels = ["Stay", "Leave"]
    values = [80, 20] if prediction[0] == 0 else [30, 70]

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    ax2.pie(values, labels=labels, autopct="%1.1f%%",
            colors=["#22c55e", "#ef4444"], startangle=90)

    ax2.set_title("Risk Analysis")

    st.pyplot(fig2)

else:
    st.info("👆🏻 Enter values and click Predict")