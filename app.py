import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SITE_IMAGE = 'https://images.unsplash.com/photo-1510130387422-82bed34b0e1d?auto=format&fit=crop&w=1200&q=80'

@st.cache_data
def load_data():
    df = pd.read_csv('beer-servings.csv')
    df = df.drop(columns=[col for col in df.columns if col.startswith('Unnamed')], errors='ignore')
    df = df.dropna(subset=['total_litres_of_pure_alcohol'])
    return df

@st.cache_resource
def train_model(df):
    X = df[['beer_servings', 'spirit_servings', 'wine_servings', 'continent']].copy()
    y = df['total_litres_of_pure_alcohol']

    preprocessor = ColumnTransformer(
        transformers=[
            ('continent', OneHotEncoder(handle_unknown='ignore'), ['continent'])
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=150, random_state=42))
        ]
    )
    pipeline.fit(X, y)
    return pipeline

@st.cache_data
def get_continent_options(df):
    continents = sorted(df['continent'].dropna().unique().tolist())
    if 'Other' not in continents:
        continents.append('Other')
    return continents


def main():
    st.set_page_config(page_title='Beer Servings Alcohol Predictor', page_icon='🍺', layout='centered')
    st.title('Beer Servings Alcohol Prediction')
    st.image(SITE_IMAGE, caption='Welcome to the Beer Servings Regression App', use_column_width=True)

    st.markdown(
        'Predict the total litres of pure alcohol per person using beer, spirit, and wine servings data. '
        'This app is trained on the Beer Servings dataset and runs a regression model inside Streamlit.'
    )

    df = load_data()
    model = train_model(df)
    continents = get_continent_options(df)

    st.sidebar.header('Input features')
    beer_servings = st.sidebar.slider('Beer servings per person', min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    spirit_servings = st.sidebar.slider('Spirit servings per person', min_value=0.0, max_value=500.0, value=50.0, step=1.0)
    wine_servings = st.sidebar.slider('Wine servings per person', min_value=0.0, max_value=500.0, value=30.0, step=1.0)
    continent = st.sidebar.selectbox('Continent', continents, index=continents.index('Europe') if 'Europe' in continents else 0)

    if st.sidebar.button('Predict total alcohol'):
        sample = pd.DataFrame([
            {
                'beer_servings': beer_servings,
                'spirit_servings': spirit_servings,
                'wine_servings': wine_servings,
                'continent': continent,
            }
        ])
        prediction = model.predict(sample)[0]
        st.success(f'Predicted total litres of pure alcohol: {prediction:.2f}')
        st.write('### Input summary')
        st.write(sample)

    st.markdown('---')
    st.write('### Training dataset sample')
    st.write(df.head(10))
    st.write('### Model features used')
    st.write('- beer_servings\n- spirit_servings\n- wine_servings\n- continent')

    st.info('To deploy this app on Streamlit Cloud, push this repository to GitHub, then connect it using the Streamlit Cloud dashboard and set the app path to `app.py`.')


if __name__ == '__main__':
    main()


import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("beer_alcohol_model.joblib")

# Title
st.title("🍺 Beer Alcohol Prediction App")

# Welcome text
st.write("Predict total litres of pure alcohol.")

# Image
image = Image.open("image.jpg")
st.image(image, use_container_width=True)

# Inputs
beer = st.number_input("Beer Servings", min_value=0)

spirit = st.number_input("Spirit Servings", min_value=0)

wine = st.number_input("Wine Servings", min_value=0)

# Predict button
if st.button("Predict"):

    input_data = np.array([[beer, spirit, wine]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Alcohol Consumption: {prediction[0]:.2f}"
    )


    