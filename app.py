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

# Assuming 'Plant' is the non-numeric column in your dataset
categorical_columns = [["Plant", "Plant_Disease_Management", "Pest_Management"]]

# Creating transformers for numeric and categorical columns
numeric_features = X_plant.select_dtypes(include=[np.number]).columns
numeric_transformer = Pipeline(
    steps=[
        (
            "num",
            SimpleImputer(strategy="median"),
        )  # You can use other imputation strategies as well
    ]
)

categorical_features = X_plant.select_dtypes(include=[np.object]).columns
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combining transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Creating the final pipeline with the RandomForestRegressor
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)
# Split the data into training and testing sets
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X_plant, y_plant, test_size=0.2, random_state=42
)

# Training the model
rf_model.fit(X_train_plant, y_train_plant)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_plant)

# Evaluate the model
rf_rmse = np.sqrt(mean_squared_error(y_test_plant, rf_predictions))
print(f"Random Forest RMSE: {rf_rmse:.2f}")

# Plant Prediction
st.header("Plant Prediction")
# Make predictions on the test set
rf_predictions = rf_model.predict(X_test_plant) 0  # You can set a default value or any appropriate value

# Sort the columns to ensure the order is the same
# plant_prediction_input = plant_prediction_input[X_train_plant.columns]



# Random Forest Model Training for Animal
st.header("Random Forest Model Training for Animal")

# For the plant dataset
X_anim = animal_df.drop('Animal_Harvest_Litres', axis=1)
y_anim = animal_df['Animal_Harvest_Litres']

# Split the data into training and testing sets
X_train_anim, X_test_anim, y_train_anim, y_test_anim = train_test_split(X_anim, y_anim, test_size=0.2, random_state=42)

# Assuming 'Animal_Harvest_Litres' is the target variable in your dataset
target_imputer = SimpleImputer(strategy='median')
y_train_anim = target_imputer.fit_transform(y_train_anim.values.reshape(-1, 1)).flatten()

# Assuming 'Animal_Group', 'Animal_Type', etc. are the categorical columns in your dataset
categorical_columns = ['Animal_Group', 'Animal_Type', 'Animal_Diseases_Management', 'Disease_Type', 'Disease_Treatment']

# Creating transformers for numeric and categorical columns
numeric_features = X_train_anim.select_dtypes(include=[np.number]).columns
numeric_transformer = Pipeline(steps=[
    ('num', SimpleImputer(strategy='median'))  # You can use other imputation strategies as well
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combining transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Creating the final pipeline with the XGBRegressor
xgb_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', XGBRegressor(random_state=42))])

# Training the model
xgb_model.fit(X_train_anim, y_train_anim)

# Make predictions on the test set
xgb_predictions = xgb_model.predict(X_test_anim)
# Replace NaN values in y_test_anim with 0
y_test_anim = y_test_anim.fillna(0)

# Evaluate the model
xgb_rmse = np.sqrt(mean_squared_error(y_test_anim, xgb_predictions))
print(f'XGBoost RMSE: {xgb_rmse:.2f}')
