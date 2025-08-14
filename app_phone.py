import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn

# Load pretrained model
MODEL_PATH = "best_phone_usage_model2.joblib"
model = joblib.load(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Phone Usage Analysis", layout="wide")
st.title("📱 Phone Usage Analysis and Prediction")

# Tabs
tabs = st.tabs(["Upload & Clean Data", "EDA & Visualization", "Prediction", "Clustering"])

# --------------------
# Tab 1: Upload & Clean Data
# --------------------
with tabs[0]:
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Loaded Successfully!")
        st.dataframe(df.head())

        # Data Cleaning
        st.subheader("Data Cleaning")
        # Example cleaning: Fill missing values
        df.fillna(method='ffill', inplace=True)
        # Standardize categorical features (example: OS, Device Model)
        for col in ['OS', 'Device_Model']:
            if col in df.columns:
                df[col] = df[col].str.lower().str.strip()
        st.success("Data Cleaning Completed")
        st.dataframe(df.head())

# --------------------
# Tab 2: EDA & Visualization
# --------------------
with tabs[1]:
    if 'df' in locals():
        st.header("Exploratory Data Analysis")

        st.subheader("Basic Statistics")
        st.write(df.describe())

        st.subheader("User Distribution by Gender")
        if 'Gender' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='Gender', data=df, ax=ax)
            st.pyplot(fig)
            mlflow.log_figure(fig, "gender_distribution.png")

        st.subheader("Device Usage Distribution")
        if 'Device_Model' in df.columns:
            top_devices = df['Device_Model'].value_counts().nlargest(10)
            fig, ax = plt.subplots(figsize=(10,5))
            sns.barplot(x=top_devices.index, y=top_devices.values, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            mlflow.log_figure(fig, "device_distribution.png")

# --------------------
# Tab 3: Prediction
# --------------------
# --------------------
# Tab 3: Prediction
# --------------------
with tabs[2]:
    if 'df' in locals():
        st.header("Predict User Primary Use")

        st.subheader("Enter Feature Values for Prediction")
        input_data = {}

        # Remove target variable from input form
        features_only = [col for col in df.columns if col.lower() != 'primary use']

        for col in features_only:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(col, df[col].unique())
            else:
                col_mean = df[col].mean()

                if pd.api.types.is_integer_dtype(df[col]) or np.all(df[col] % 1 == 0):
                    # Integer columns (Age, Number of Apps Installed, Calls Duration)
                    input_data[col] = st.number_input(col, value=int(col_mean), step=1)
                elif "hrs/day" in col.lower():
                    # Time columns (one decimal place)
                    input_data[col] = st.number_input(col, value=float(round(col_mean, 1)), step=0.1, format="%.1f")
                elif "inr" in col.lower() or "cost" in col.lower() or "spend" in col.lower():
                    # Currency columns (two decimal places)
                    input_data[col] = st.number_input(col, value=float(round(col_mean, 2)), step=1.0, format="%.2f")
                else:
                    # General float columns
                    input_data[col] = st.number_input(col, value=float(round(col_mean, 2)), step=0.01, format="%.2f")

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables if needed
    for col in input_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)

    # If prediction is numeric and model has classes_, map it
    if hasattr(model, 'classes_') and np.issubdtype(type(prediction[0]), np.integer):
        predicted_label = model.classes_[prediction[0]]
    else:
        predicted_label = prediction[0]

    st.success(f"Predicted User Primary Use: {predicted_label}")




# --------------------
# Tab 4: Clustering
# --------------------
with tabs[3]:
    if 'df' in locals():
        st.header("Clustering Users")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols])

            # PCA for 2D visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # KMeans clustering
            k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_pca)

            df['Cluster'] = clusters
            fig, ax = plt.subplots()
            sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2', ax=ax)
            st.pyplot(fig)
            mlflow.log_figure(fig, "user_clusters.png")
        else:
            st.warning("Not enough numeric features for clustering")