import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

model = joblib.load(
    "models/random_forest_model.joblib"
)

scaler = joblib.load(
    "models/scaler.joblib"
)

with open("models/cluster_names.json", "r", encoding="utf-8") as f:
    cluster_names = json.load(f)

st.title("SegmentIQ")
st.subheader("Prediction du segment client")

recency = st.number_input(
    "Recency",
    min_value=0,
    value=10
)

frequency = st.number_input(
    "Frequency",
    min_value=1,
    value=8
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0,
    value=5000.0
)

predict_button = st.button("Predire le segment")

if predict_button:
    new_customer = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    new_customer_log = np.log1p(new_customer)

    new_customer_scaled = scaler.transform(
        new_customer_log
    )

    predicted_cluster = model.predict(
        new_customer_scaled
    )

    cluster = predicted_cluster[0]

    segment = cluster_names[str(cluster)]

    st.success(f"Segment : {segment}")

    st.write(f"Cluster predit : {cluster}")

    st.write("### Informations du client")

    st.write(f"Recency : {recency}")
    st.write(f"Frequency : {frequency}")
    st.write(f"Monetary : {monetary}")