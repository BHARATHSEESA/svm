import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# ---------------- TITLE ----------------

st.title("SVM Iris Flower Prediction App")

st.write("Support Vector Machine Classification using Different Kernels")

# ---------------- LOAD DATA ----------------

iris = load_iris()

X = iris.data
y = iris.target

# ---------------- SPLIT DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- KERNEL SELECTION ----------------

kernel = st.selectbox(
    "Select SVM Kernel",
    ("linear", "rbf", "poly", "sigmoid")
)

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------

model = SVC(
    kernel=kernel,
    C=1.0,
    gamma='scale'
)

# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)

# ---------------- ACCURACY ----------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.success(f"{kernel.upper()} Kernel Accuracy: {accuracy:.2f}")

# ---------------- USER INPUT ----------------

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)

# ---------------- PREDICT BUTTON ----------------

if st.button("Predict Flower"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # SCALE INPUT DATA
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    flower = iris.target_names[prediction[0]]

    st.subheader("Prediction Result")

    st.success(f"Predicted Flower: {flower}")

# ---------------- KERNEL COMPARISON ----------------

st.subheader("Kernel Comparison")

kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for k in kernels:

    temp_model = SVC(
        kernel=k,
        C=1.0,
        gamma='scale'
    )

    temp_model.fit(X_train, y_train)

    temp_pred = temp_model.predict(X_test)

    temp_accuracy = accuracy_score(y_test, temp_pred)

    st.write(f"{k.upper()} Kernel Accuracy : {temp_accuracy:.2f}")
