import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit webpage setup
st.title('Spotify Stream Predictor')

# User inputs
bpm = st.number_input('Enter BPM:', min_value=0, max_value=300, value=120)
danceability = st.number_input('Danceability (0-100):', min_value=0, max_value=100, value=50)
valence = st.number_input('Valence (0-100):', min_value=0, max_value=100, value=50)
energy = st.number_input('Energy (0-100):', min_value=0, max_value=100, value=50)
acousticness = st.number_input('Acousticness (0-100):', min_value=0, max_value=100, value=50)
liveness = st.number_input('Liveness (0-100):', min_value=0, max_value=100, value=50)
speechiness = st.number_input('Speechiness (0-100):', min_value=0, max_value=100, value=50)

# Predict button
if st.button('Predict'):
    # Feature array
    features = np.array([[bpm, danceability, valence, energy, acousticness, liveness, speechiness]])
    # Scaling features
    features_scaled = scaler.transform(features)
    
    # Prediction
    prediction = model.predict(features_scaled)

    formatted_prediction = "{:,}".format(int(prediction[0]))
    st.write(f'Estimated Streams: {formatted_prediction}')




#https://docs.streamlit.io/streamlit-community-cloud/get-started/quickstart
#docu to get deploying