import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load data
plant_df = pd.read_csv("plant_data.csv")
animal_df = pd.read_csv("animal_data.csv")

# Sidebar for user input
st.sidebar.title("User Input")

# Plant Prediction Section
st.sidebar.subheader("Plant Prediction")
plant_input = {}
# Assuming 'Plant', 'Plant_Disease_Management', 'Pest_Management', etc. are the features in your dataset
plant_input["Plant"] = st.sidebar.selectbox("Select Plant", plant_df["Plant"].unique())
plant_input["Plant_Disease_Management"] = st.sidebar.selectbox(
    "Disease Management", plant_df["Plant_Disease_Management"].unique()
)
plant_input["Pest_Management"] = st.sidebar.selectbox(
    "Pest Management", plant_df["Pest_Management"].unique()
)
# Add more input options as needed

# Animal Prediction Section
st.sidebar.subheader("Animal Prediction")
animal_input = {}
# Assuming 'Animal_Group', 'Animal_Type', 'Animal_Diseases_Management', 'Disease_Type', etc. are the features in your dataset
animal_input["Animal_Group"] = st.sidebar.selectbox(
    "Select Animal Group", animal_df["Animal_Group"].unique()
)
animal_input["Animal_Type"] = st.sidebar.selectbox(
    "Select Animal Type", animal_df["Animal_Type"].unique()
)
animal_input["Animal_Diseases_Management"] = st.sidebar.selectbox(
    "Diseases Management", animal_df["Animal_Diseases_Management"].unique()
)
# Add more input options as needed

# Main content
st.title("Plant and Animal Data Analysis App")
# Debugging
st.write("Before creating preprocessor")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), ["numeric_column"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["categorical_column"]),
    ]
)
st.write("After creating preprocessor")

# Creating the final pipeline with the RandomForestRegressor
st.write("Before creating rf_model_plant")
rf_model_plant = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)
st.write("After creating rf_model_plant")
# Plant Model Training
st.header("Random Forest Model Training for Plant")

# Assuming 'target_column_plant' is the target variable in your plant dataset
X_plant = plant_df.drop("Plant_Harvest_Kg", axis=1)
y_plant = plant_df["Plant_Harvest_Kg"]

# Split the data into training and testing sets
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X_plant, y_plant, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf_model_plant = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Training the model
st.write("Before fitting rf_model_plant")
rf_model_plant.fit(X_train_plant, y_train_plant)
st.write("After fitting rf_model_plant")

# Make predictions on the test set
rf_predictions_plant = rf_model_plant.predict(X_test_plant)

# Evaluate the model
rf_rmse_plant = np.sqrt(mean_squared_error(y_test_plant, rf_predictions_plant))
st.subheader("Random Forest Model Evaluation for Plant")
st.write(f"Random Forest RMSE: {rf_rmse_plant:.2f}")

# Animal Model Training
st.header("Random Forest Model Training for Animal")

# Assuming 'target_column_animal' is the target variable in your animal dataset
X_anim = animal_df.drop("Animal_Harvest_Litres", axis=1)
y_anim = animal_df["Animal_Harvest_Litres"]

# Split the data into training and testing sets
X_train_anim, X_test_anim, y_train_anim, y_test_anim = train_test_split(
    X_anim, y_anim, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf_model_anim = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

rf_model_anim.fit(X_train_anim, y_train_anim)

# Make predictions on the test set
rf_predictions_anim = rf_model_anim.predict(X_test_anim)

# Evaluate the model
rf_rmse_anim = np.sqrt(mean_squared_error(y_test_anim, rf_predictions_anim))
st.subheader("Random Forest Model Evaluation for Animal")
st.write(f"Random Forest RMSE: {rf_rmse_anim:.2f}")

# Plant Prediction
st.header("Plant Prediction")
plant_prediction_input = pd.DataFrame([plant_input])

# Predict Plant Harvest
plant_prediction = rf_model_plant.predict(plant_prediction_input)
st.write(f"Predicted Plant Harvest (Kg): {plant_prediction[0]:.2f}")

# Animal Prediction
st.header("Animal Prediction")
animal_prediction_input = pd.DataFrame([animal_input])

# Predict Animal Harvest Litres
animal_prediction = rf_model_anim.predict(animal_prediction_input)
st.write(f"Predicted Animal Harvest Litres: {animal_prediction[0]:.2f}")

# Show the app
st.sidebar.text(
    "Note: Check the options in the sidebar to input data and get predictions."
)
