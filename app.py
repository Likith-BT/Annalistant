import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title='AI Analytics Assistant', layout='wide')

st.title("📊 AI Analytics Assistant")
st.write("Upload your dataset and explore insights with visualization and machine learning models.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Basic Info
    # -------------------------------
    st.subheader("Dataset Information")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # -------------------------------
    # Basic Cleaning
    # -------------------------------
    def basic_cleaning(df):
        df = df.dropna()
        df = df.drop_duplicates()
        return df

    if st.button("Clean Data"):
        df = basic_cleaning(df)
        st.success("✅ Data cleaned (missing values & duplicates removed).")
        st.write("New shape:", df.shape)

    # -------------------------------
    # Data Visualization
    # -------------------------------
    st.subheader("Data Visualization")

    column = st.selectbox("Select a column for visualization", df.columns)
    if df[column].dtype in ['int64', 'float64']:
        st.bar_chart(df[column].value_counts())
    else:
        fig, ax = plt.subplots()
        sns.countplot(x=df[column], ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Machine Learning
    # -------------------------------
    st.subheader("Machine Learning (Classification)")

    target_column = st.selectbox("Select target column", df.columns)

    if st.button("Run ML Model"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical variables to numeric
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("This is an AI-powered analytics assistant built with Streamlit, pandas, scikit-learn, matplotlib, and seaborn.")
st.sidebar.write("Upload data → Clean → Explore → Build ML models in one app.")
st.sidebar.markdown("---")
st.sidebar.write("Built by Likith BT 🚀")
