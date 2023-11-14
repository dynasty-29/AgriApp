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

# Plant Model Training
st.header("Random Forest Model Training for Plant")

# Assuming 'Plant' is the non-numeric column in your dataset
categorical_columns_plant = ["Plant", "Plant_Disease_Management", "Pest_Management"]

# Creating transformers for numeric and categorical columns
numeric_features_plant = plant_df.select_dtypes(include=[np.number]).columns
numeric_transformer_plant = Pipeline(steps=[("num", SimpleImputer(strategy="median"))])

categorical_transformer_plant = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combining transformers
preprocessor_plant = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_plant, numeric_features_plant),
        ("cat", categorical_transformer_plant, categorical_columns_plant),
    ]
)

# Creating the final pipeline with the RandomForestRegressor
rf_model_plant = Pipeline(
    steps=[
        ("preprocessor", preprocessor_plant),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Split the data into training and testing sets
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    plant_df.drop("Plant_Harvest_Kg", axis=1),
    plant_df["Plant_Harvest_Kg"],
    test_size=0.2,
    random_state=42,
)

# Training the model
rf_model_plant.fit(X_train_plant, y_train_plant)
## Plant Prediction
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

# Random Forest Model Training for Animal
st.header("Random Forest Model Training for Animal")

# Assuming 'Animal_Group', 'Animal_Type', etc. are the categorical columns in your dataset
categorical_columns_anim = [
    "Animal_Group",
    "Animal_Type",
    "Animal_Diseases_Management",
    "Disease_Type",
    "Disease_Treatment",
]

# Creating transformers for numeric and categorical columns
numeric_features_anim = animal_df.select_dtypes(include=[np.number]).columns
numeric_transformer_anim = Pipeline(steps=[("num", SimpleImputer(strategy="median"))])

categorical_transformer_anim = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combining transformers
preprocessor_anim = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_anim, numeric_features_anim),
        ("cat", categorical_transformer_anim, categorical_columns_anim),
    ]
)

# Creating the final pipeline with the RandomForestRegressor
rf_model_anim = Pipeline(
    steps=[
        ("preprocessor", preprocessor_anim),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

# Split the data into training and testing sets
X_train_anim, X_test_anim, y_train_anim, y_test_anim = train_test_split(
    animal_df.drop("Animal_Harvest_Litres", axis=1),
    animal_df["Animal_Harvest_Litres"],
    test_size=0.2,
    random_state=42,
)

# Training the model
rf_model_anim.fit(X_train_anim, y_train_anim)

# Make predictions on the test set
rf_predictions_anim = rf_model_anim.predict(X_test_anim)

# Evaluate the model
rf_rmse_anim = np.sqrt(mean_squared_error(y_test_anim, rf_predictions_anim))
st.write(f"Random Forest RMSE for Animal Dataset: {rf_rmse_anim:.2f}")

# Animal Prediction
st.header("Animal Prediction")

# Animal prediction input from sidebar
animal_prediction_input = pd.DataFrame([animal_input])

# Check if the columns match before transforming
if list(animal_prediction_input.columns) == list(X_train_anim.columns):
    # Try to transform the input using the preprocessor
    transformed_animal_prediction_input = preprocessor_anim.transform(
        animal_prediction_input
    )
    st.write("After transforming animal_prediction_input")
    st.write("Shape after transform:", transformed_animal_prediction_input.shape)
else:
    st.write(
        "Columns in animal_prediction_input do not match X_train_anim. Please check your input."
    )

# Make predictions
animal_prediction = rf_model_anim.predict(transformed_animal_prediction_input)
st.write(f"Predicted Animal Harvest (Litres): {animal_prediction[0]:.2f}")
