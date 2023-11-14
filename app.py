import streamlit as st
import pandas as pd

# Load data
plant_df = pd.read_csv("plant_data.csv")
animal_df = pd.read_csv("animal_data.csv")

# Sidebar for user input
st.sidebar.title("User Input")

# Plant Prediction Section in the sidebar
st.sidebar.subheader("Plant Prediction")
plant_input = {}

# Assuming 'Plant_Disease_Management', 'Pest_Management', etc. are the features in your dataset
plant_input["Plant"] = st.sidebar.selectbox(
    "Plant", plant_df["Plant"].unique(), key="plant_selectbox"
)
plant_input["Plant_Disease_Management"] = st.sidebar.selectbox(
    "Disease Management",
    plant_df["Plant_Disease_Management"].unique(),
    key="disease_management_selectbox",
)
plant_input["Pest_Management"] = st.sidebar.selectbox(
    "Pest Management",
    plant_df["Pest_Management"].unique(),
    key="pest_management_selectbox",
)

# Display user input for plant prediction
st.subheader("User Input for Plant Prediction:")
st.write(plant_input)

# Display predicted value (placeholder, replace it with your prediction logic)
st.subheader("Predicted Plant Harvest:")
# st.write("Replace this with your prediction logic")
plant_input["Plant_Harvest_Kg"] = st.sidebar.selectbox(
    "Pest Management",
    plant_df["Plant_Harvest_Kg"].unique(),
    key="plant_Harvest_Kg",
)

# Animal Prediction Section in the sidebar
st.sidebar.subheader("Animal Prediction")
animal_input = {}

# Assuming 'Animal_Group', 'Animal_Type', etc. are the features in your dataset
animal_input["Animal_Group"] = st.sidebar.selectbox(
    "Animal Group", animal_df["Animal_Group"].unique(), key="animal_group_selectbox"
)
animal_input["Animal_Type"] = st.sidebar.selectbox(
    "Animal Type", animal_df["Animal_Type"].unique(), key="animal_type_selectbox"
)
animal_input["Animal_Diseases_Management"] = st.sidebar.selectbox(
    "Diseases Management",
    animal_df["Animal_Diseases_Management"].unique(),
    key="animal_disease_management_selectbox",
)
animal_input["Disease_Type"] = st.sidebar.selectbox(
    "Disease", animal_df["Disease_Type"].unique(), key="disease_type_selectbox"
)
animal_input["Disease_Treatment"] = st.sidebar.selectbox(
    "Diseases treatment",
    animal_df["Disease_Treatment"].unique(),
    key="disease_treatment_selectbox",
)

# Display user input for animal prediction
st.subheader("User Input for Animal Prediction:")
st.write(animal_input)

# Display predicted value (placeholder, replace it with your prediction logic)
st.subheader("Predicted Animal Harvest:")
# st.write("Replace this with your prediction logic")
animal_input["Animal_Harvest_Litres"] = st.selectbox(
    "Diseases treatment",
    animal_df["Animal_Harvest_Litres"].unique(),
    key="animal_Harvest_Litres_selectbox",
)
