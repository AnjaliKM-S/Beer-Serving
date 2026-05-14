import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Beer Servings Alcohol Prediction",
    layout="wide"
)

# -----------------------------
# Welcome Image
# -----------------------------
SITE_IMAGE = "https://images.unsplash.com/photo-1510130387422-82bed34b0e1d?auto=format&fit=crop&w=1200&q=80"

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("beer-servings.csv")

    df = df.drop(
        columns=[col for col in df.columns if col.startswith("Unnamed")],
        errors="ignore"
    )

    df = df.dropna(
        subset=["total_litres_of_pure_alcohol"]
    )

    return df

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(df):

    X = df[
        [
            "beer_servings",
            "spirit_servings",
            "wine_servings",
            "continent"
        ]
    ]

    y = df["total_litres_of_pure_alcohol"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "continent",
                OneHotEncoder(handle_unknown="ignore"),
                ["continent"]
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=150,
                random_state=42
            ))
        ]
    )

    pipeline.fit(X, y)

    return pipeline

# -----------------------------
# Continent Options
# -----------------------------
@st.cache_data
def get_continent_options(df):

    continents = sorted(
        df["continent"].dropna().unique().tolist()
    )

    return continents

# -----------------------------
# Main App
# -----------------------------
df = load_data()

model = train_model(df)

continents = get_continent_options(df)

# -----------------------------
# Title & Image
# -----------------------------
st.title("🍺 Beer Servings Alcohol Prediction")

st.write(
    "Predict total litres of pure alcohol using beer, spirit, wine servings and continent."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Features")

beer_servings = st.sidebar.slider(
    "Beer Servings",
    min_value=0,
    max_value=500,
    value=100
)

spirit_servings = st.sidebar.slider(
    "Spirit Servings",
    min_value=0,
    max_value=500,
    value=50
)

wine_servings = st.sidebar.slider(
    "Wine Servings",
    min_value=0,
    max_value=500,
    value=30
)

continent = st.sidebar.selectbox(
    "Continent",
    continents
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Total Alcohol"):

    sample = pd.DataFrame({
        "beer_servings": [beer_servings],
        "spirit_servings": [spirit_servings],
        "wine_servings": [wine_servings],
        "continent": [continent]
    })

    prediction = model.predict(sample)[0]

    st.success(
        f"Predicted Total Litres of Pure Alcohol: {prediction:.2f}"
    )

    st.write("### Input Summary")

    st.dataframe(sample)




import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("beer_alcohol_model.joblib")


# Image
image = Image.open("image.jpg")
st.image(image, use_container_width=True)

# Inputs
beer = st.slider("Beer Servings", min_value=0,
    max_value=500,
    value=50)

spirit = st.slider("Spirit Servings", min_value=0,
    max_value=500,
    value=50)

wine = st.slider("Wine Servings", min_value=0,
    max_value=500,
    value=50)

# Predict button
if st.button("Predict"):

    input_data = np.array([[beer, spirit, wine]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Alcohol Consumption: {prediction[0]:.2f}"
    )


    