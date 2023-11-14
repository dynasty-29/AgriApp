import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
plant_df = pd.read_csv("plant_data.csv")
animal_df = pd.read_csv("animal_data.csv")

# Sidebar for user input
st.sidebar.title("User Input")

# Plant Prediction Section
st.sidebar.subheader("Plant Prediction")
plant_input = {}

# Assuming 'Plant_Disease_Management', 'Pest_Management', etc. are the features in your dataset
plant_input["Plant"] = st.sidebar.selectbox("Plant", plant_df["Plant"].unique())
plant_input["Plant_Disease_Management"] = st.sidebar.selectbox(
    "Disease Management", plant_df["Plant_Disease_Management"].unique()
)
plant_input["Pest_Management"] = st.sidebar.selectbox(
    "Pest Management", plant_df["Pest_Management"].unique()
)

# Animal Prediction Section
st.sidebar.subheader("Animal Prediction")
animal_input = {}

# Assuming 'Animal_Group', 'Animal_Type', etc. are the features in your dataset
animal_input["Animal_Group"] = st.sidebar.selectbox(
    "Animal Group", animal_df["Animal_Group"].unique()
)
animal_input["Animal_Type"] = st.sidebar.selectbox(
    "Animal Type", animal_df["Animal_Type"].unique()
)
animal_input["Animal_Diseases_Management"] = st.sidebar.selectbox(
    "Diseases Management", animal_df["Animal_Diseases_Management"].unique()
)
animal_input["Disease_Type"] = st.sidebar.selectbox(
    "Disease", animal_df["Disease_Type"].unique()
)
animal_input["Disease_Treatment"] = st.sidebar.selectbox(
    "Diseases treatment", animal_df["Disease_Treatment"].unique()
)

# Plant Prediction Section
st.header("Plant Prediction")

# Plant Prediction input from sidebar
plant_prediction_input = {}

# Assuming 'Plant_Disease_Management', 'Pest_Management', etc. are the features in your dataset
plant_prediction_input["Plant"] = st.sidebar.selectbox(
    "Plant", plant_df["Plant"].unique()
)
plant_prediction_input["Plant_Disease_Management"] = st.sidebar.selectbox(
    "Disease Management", plant_df["Plant_Disease_Management"].unique()
)
plant_prediction_input["Pest_Management"] = st.sidebar.selectbox(
    "Pest Management", plant_df["Pest_Management"].unique()
)

# Display user input for plant prediction
st.subheader("User Input for Plant Prediction:")
st.write(plant_prediction_input)

# Display predicted value (placeholder, replace it with your prediction logic)
st.subheader("Predicted Plant Harvest:")
st.write("Replace this with your prediction logic")

# Animal Prediction Section
st.header("Animal Prediction")

# Animal Prediction input from sidebar
animal_prediction_input = {}

# Assuming 'Animal_Group', 'Animal_Type', etc. are the features in your dataset
animal_prediction_input["Animal_Group"] = st.sidebar.selectbox(
    "Animal Group", animal_df["Animal_Group"].unique()
)
animal_prediction_input["Animal_Type"] = st.sidebar.selectbox(
    "Animal Type", animal_df["Animal_Type"].unique()
)
animal_prediction_input["Animal_Diseases_Management"] = st.sidebar.selectbox(
    "Diseases Management", animal_df["Animal_Diseases_Management"].unique()
)
animal_prediction_input["Disease_Type"] = st.sidebar.selectbox(
    "Disease", animal_df["Disease_Type"].unique()
)
animal_prediction_input["Disease_Treatment"] = st.sidebar.selectbox(
    "Diseases treatment", animal_df["Disease_Treatment"].unique()
)

# Display user input for animal prediction
st.subheader("User Input for Animal Prediction:")
st.write(animal_prediction_input)

# Display predicted value (placeholder, replace it with your prediction logic)
st.subheader("Predicted Animal Harvest:")
st.write("Replace this with your prediction logic")
