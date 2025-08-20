import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import shap

# ===== Load Data =====
@st.cache_data
def load_data():
    # Replace with the dataset you used
    df = pd.read_csv("fraud_dataset.csv")  
    return df

# ===== Load Model =====
@st.cache_resource
def load_model():
    rf_model = joblib.load("random_forest_model.pkl")
    return rf_model

# ===== Main App =====
def main():
    st.title("🔍 Fraud Detection Dashboard")
    st.write("An interactive dashboard to explore fraud detection insights.")

    # Sidebar navigation
    menu = ["Dataset Overview", "Model Evaluation", "Explainability (SHAP)"]
    choice = st.sidebar.radio("Go to", menu)

    df = load_data()

    if choice == "Dataset Overview":
        st.subheader("Dataset Overview")
        st.write(df.head())

        # Class distribution
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='class', ax=ax)
        st.pyplot(fig)

        # Purchase value distribution
        st.subheader("Purchase Value Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['purchase_value'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    elif choice == "Model Evaluation":
        st.subheader("Model Evaluation")

        # Load model
        rf_model = load_model()

        # Predict on features
        X = df.drop("class", axis=1)
        y = df["class"]
        y_pred = rf_model.predict(X)
        y_proba = rf_model.predict_proba(X)[:, 1]

        # Metrics
        st.text("Classification Report:")
        st.text(classification_report(y, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        st.write(cm)

        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y, y_proba)
        avg_precision = average_precision_score(y, y_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f"AP={avg_precision:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)

    elif choice == "Explainability (SHAP)":
        st.subheader("Model Explainability with SHAP")

        rf_model = load_model()
        X = df.drop("class", axis=1)

        # Use small sample for SHAP
        X_sample = X.sample(100, random_state=42)

        explainer = shap.Explainer(rf_model, X_sample)
        shap_values = explainer(X_sample)

        st.write("### Feature Importance (SHAP Bar Plot)")
        fig = shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

        st.write("### Beeswarm Plot")
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

# Run app
if __name__ == "__main__":
    main()
