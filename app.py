import streamlit as st
import pandas as pd
import numpy as np
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

# Assuming 'Animal_Diseases_Management', 'Disease_Type', etc. are the features in your dataset
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

# Main content
st.title("Plant and Animal Data Analysis App")

# Plant Model Training
st.header("Random Forest Model Training for Plant")

# Assuming 'target_column_plant' is the target variable in your plant dataset
X_plant = plant_df.drop("Plant_Harvest_Kg", axis=1)
y_plant = plant_df["Plant_Harvest_Kg"]

# Split the data into training and testing sets
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X_plant, y_plant, test_size=0.2, random_state=42
)

# Assuming 'categorical_columns_plant' is a list of categorical columns in your plant dataset
categorical_columns_plant = ["Plant", "Plant_Disease_Management", "Pest_Management"]

# Creating transformers for numeric and categorical columns for the plant dataset
numeric_features_plant = X_train_plant.select_dtypes(include=[np.number]).columns
numeric_transformer_plant = Pipeline(steps=[("num", SimpleImputer(strategy="median"))])

categorical_transformer_plant = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combining transformers for the plant dataset
preprocessor_plant = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_plant, numeric_features_plant),
        ("cat", categorical_transformer_plant, categorical_columns_plant),
    ]
)

# Creating the final pipeline with the RandomForestRegressor for the plant dataset
rf_model_plant = Pipeline(
    steps=[
        ("preprocessor", preprocessor_plant),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

rf_model_plant.fit(X_train_plant, y_train_plant)
#
# Plant Prediction
st.header("Plant Prediction")
plant_prediction_input = pd.DataFrame([plant_input])

# Ensure that plant_prediction_input has the same columns as X_train_plant
missing_columns = set(X_train_plant.columns) - set(plant_prediction_input.columns)
for column in missing_columns:
    plant_prediction_input[
        column
    ] = 0  # You can set a default value or any appropriate value

# Sort the columns to ensure the order is the same
plant_prediction_input = plant_prediction_input[X_train_plant.columns]

# Try to transform the input using the preprocessor and catch any exceptions
try:
    transformed_plant_prediction_input = preprocessor_plant.transform(
        plant_prediction_input.iloc[0:1]  # Using iloc to get a Series
    )
    st.write("After transforming plant_prediction_input")
    st.write("Columns after transform:", transformed_plant_prediction_input.columns)
    st.write("Shape after transform:", transformed_plant_prediction_input.shape)
except Exception as e:
    st.write(f"Error during transformation: {e}")
    transformed_plant_prediction_input = None

# Check if transformation was successful before predicting
if transformed_plant_prediction_input is not None:
    # Predict Plant Harvest
    plant_prediction = rf_model_plant.predict(transformed_plant_prediction_input)
    st.write(f"Predicted Plant Harvest (Kg): {plant_prediction[0]:.2f}")
else:
    st.write("Transformation failed. Please check your input.")

# Random Forest Model Training for Animal
st.header("Random Forest Model Training for Animal")

# Define rf_model_anim before training
rf_model_anim = Pipeline(
    steps=[
        ("preprocessor", preprocessor_anim),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Assuming 'target_column_animal' is the target variable in your animal dataset
X_anim = animal_df.drop("Animal_Harvest_Litres", axis=1)
y_anim = animal_df["Animal_Harvest_Litres"]

# Split the data into training and testing sets
X_train_anim, X_test_anim, y_train_anim, y_test_anim = train_test_split(
    X_anim, y_anim, test_size=0.2, random_state=42
)

# Debugging information
st.write("Before transforming X_train_anim")
st.write("Columns before transform:", X_train_anim.columns)
st.write("Shape before transform:", X_train_anim.shape)

# Check for NaN or infinite values in numeric columns of X_train_anim
numeric_cols_anim = X_train_anim.select_dtypes(include=[np.number]).columns
nan_inf_counts_anim = X_train_anim[numeric_cols_anim].apply(
    lambda x: x.isnull().sum() + np.isinf(x).sum()
)

st.write(
    "NaN or infinite values in X_train_anim (numeric columns only):",
    nan_inf_counts_anim.sum(),
)

# Train the Random Forest model
try:
    rf_model_anim.fit(X_train_anim, y_train_anim)
except Exception as e:
    st.write(f"Error during training: {e}")

# After transforming X_train_anim
st.write("After transforming X_train_anim")
st.write("Columns after transform:", X_train_anim.columns)
st.write("Shape after transform:", X_train_anim.shape)
