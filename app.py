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

# Plant Model Training
st.header("Random Forest Model Training for Plant")

# Assuming 'target_column_plant' is the target variable in your plant dataset
X_plant = plant_df.drop("Plant_Harvest_Kg", axis=1)
y_plant = plant_df["Plant_Harvest_Kg"]

# Split the data into training and testing sets
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X_plant, y_plant, test_size=0.2, random_state=42
)

# Creating transformers for numeric and categorical columns
numeric_features_plant = X_train_plant.select_dtypes(include=[np.number]).columns
categorical_features_plant = [
    col for col in X_train_plant.columns if X_train_plant[col].dtype == "object"
]

if not numeric_features_plant.empty:
    numeric_transformer_plant = Pipeline(
        steps=[
            (
                "num",
                SimpleImputer(strategy="median"),
            )  # You can use other imputation strategies as well
        ]
    )
else:
    st.error("No numeric features found in the plant dataset.")
    st.stop()

if categorical_features_plant:
    categorical_transformer_plant = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
else:
    st.error("No categorical features found in the plant dataset.")
    st.stop()

# Combining transformers
preprocessor_plant = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_plant, numeric_features_plant),
        ("cat", categorical_transformer_plant, categorical_features_plant),
    ]
)

# Creating the final pipeline with the RandomForestRegressor
rf_model_plant = Pipeline(
    steps=[
        ("preprocessor", preprocessor_plant),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Training the model
rf_model_plant.fit(X_train_plant, y_train_plant)

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

# Creating transformers for numeric and categorical columns
numeric_features_anim = X_train_anim.select_dtypes(include=[np.number]).columns
numeric_transformer_anim = Pipeline(
    steps=[
        (
            "num",
            SimpleImputer(strategy="median"),
        )  # You can use other imputation strategies as well
    ]
)

categorical_features_anim = X_train_anim.select_dtypes(include=[np.object]).columns
categorical_transformer_anim = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combining transformers
preprocessor_anim = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_anim, numeric_features_anim),
        ("cat", categorical_transformer_anim, categorical_features_anim),
    ]
)

# Creating the final pipeline with the RandomForestRegressor
rf_model_anim = Pipeline(
    steps=[
        ("preprocessor", preprocessor_anim),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Training the model
rf_model_anim.fit(X_train_anim, y_train_anim)

# Make predictions on the test set
rf_predictions_anim = rf_model_anim.predict(X_test_anim)

# Evaluate the model
rf_rmse_anim = np.sqrt(mean_squared_error(y_test_anim, rf_predictions_anim))
st.subheader("Random Forest Model Evaluation for Animal")
st.write(f"Random Forest RMSE: {rf_rmse_anim:.2f}")

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
