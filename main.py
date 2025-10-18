# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import warnings

# ------------------------------
# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------
# Page Config
st.set_page_config(page_title="Breast Cancer SVM", layout="wide")

# ------------------------------
st.title("🩺 Breast Cancer Classification with SVM")
st.markdown("""
This app demonstrates **SVM** classification on the Breast Cancer Wisconsin Dataset.
- You can choose the SVM kernel: **Linear** or **RBF**
- See evaluation metrics and confusion matrix
- **Predict for a new patient** by entering feature values
- **OR choose a real sample** (Benign/Malignant) to test the model
""")

# ------------------------------
# Load Data
@st.cache_data
def load_data():
    cancer = datasets.load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    df['target'] = df['target'].map({0: 'malignant', 1: 'benign'})
    return df, cancer.feature_names

df, feature_names = load_data()

# ------------------------------
# Sidebar
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show raw data")
kernel = st.sidebar.radio("Choose SVM Kernel", ('linear', 'rbf'))

# Display raw data
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Target distribution:")
    st.bar_chart(df['target'].value_counts())

# ------------------------------
# Train-Test Split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Train SVM
svm_model = SVC(kernel=kernel, random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# ------------------------------
# Display Results
st.subheader(f"SVM Results with {kernel.upper()} Kernel")
st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))

st.subheader("Classification Report")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
st.dataframe(report_df)

# ------------------------------
# Sample Data Selection
st.subheader(" Choose a Sample Data (Optional)")
sample_choice = st.selectbox(
    "Select a sample type",
    ["None", "Benign", "Malignant"]
)

sample_data = None
if sample_choice != "None":
    # Randomly pick a sample row from the selected class
    sample_data = df[df['target'].str.lower() == sample_choice.lower()].sample(1, random_state=np.random.randint(1000))
    st.write(f"✅ Sample {sample_choice} Patient Data:")
    st.dataframe(sample_data)

# ------------------------------
# User Input for Prediction
st.subheader(" Predict for a New Patient")

with st.form("prediction_form"):
    st.write("Enter the patient's feature values (or use the sample above):")
    user_input = {}
    cols = st.columns(3)

    for idx, feature in enumerate(feature_names):
        default_val = float(sample_data[feature].values[0]) if sample_data is not None else float(df[feature].mean())
        val = cols[idx % 3].number_input(feature, value=default_val)
        user_input[feature] = val

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure columns match training data and type is float
    input_df = input_df[X.columns].astype(float)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = svm_model.predict(input_scaled)[0]

    # Display result
    st.success(f"The model predicts: **{prediction.upper()}**")
    if prediction == "malignant":
        st.warning("⚠️ This patient may have cancer. Please consult a doctor.")
    else:
        st.info("✅ This patient is predicted as benign (no cancer).")
