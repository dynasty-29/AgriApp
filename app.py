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

# Plant Prediction
st.header("Plant Prediction")

# Plant Prediction input from sidebar
plant_prediction_input = pd.DataFrame([plant_input])

# Ensure the columns are in the correct order
plant_prediction_input = plant_prediction_input[X_train_plant.columns]

# Impute missing values with median
plant_prediction_input = plant_prediction_input.apply(
    lambda x: x.fillna(x.median()) if x.dtype.kind in "biufc" else x
)

# One-hot encode categorical columns
plant_prediction_input = pd.get_dummies(plant_prediction_input)

# Check if the columns match X_train_plant
if set(plant_prediction_input.columns) == set(X_train_plant.columns):
    # Try to predict Plant Harvest
    try:
        plant_prediction = rf_model_plant.predict(plant_prediction_input)
        st.write(f"Predicted Plant Harvest: {plant_prediction[0]:.2f}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
else:
    st.write(
        "Columns in plant_prediction_input do not match X_train_plant. Please check your input."
    )

# Animal Prediction
st.header("Animal Prediction")

# Animal prediction input from sidebar
animal_prediction_input = pd.DataFrame([animal_input])

# Ensure the columns are in the correct order
animal_prediction_input = animal_prediction_input[X_train_anim.columns]

# Impute missing values with median
animal_prediction_input = animal_prediction_input.apply(
    lambda x: x.fillna(x.median()) if x.dtype.kind in "biufc" else x
)

# One-hot encode categorical columns
animal_prediction_input = pd.get_dummies(animal_prediction_input)

# Check if the columns match X_train_anim
if set(animal_prediction_input.columns) == set(X_train_anim.columns):
    # Try to predict Animal Harvest
    try:
        animal_prediction = rf_model_anim.predict(animal_prediction_input)
        st.write(f"Predicted Animal Harvest: {animal_prediction[0]:.2f}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")
else:
    st.write(
        "Columns in animal_prediction_input do not match X_train_anim. Please check your input."
    )
