#%%writefile app.py
import streamlit as st
import joblib
import pandas as pd
import os

# Set up the Streamlit app title and description
st.title("ANXPHENOTYPE Prediction for Females")
st.write("Enter the patient's information to predict the ANXPHENOTYPE.")

# Define the file path to the saved model
model_file_path = 'lr_model_female.joblib'

# Load the model with error handling
model = None
try:
    model = joblib.load(model_file_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {model_file_path}. Please ensure the model is trained and saved.")
    st.stop() # Stop execution if the model is not found
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# If the model is loaded, create input widgets for the user
if model is not None:
    st.sidebar.header("Patient Information")

    # Input widgets for features (excluding GENERO_1 as this app is for females)
    # Based on the one-hot encoded columns from X_female_train
    lte12 = st.sidebar.slider("LTE12", 0, 2, 1)

    st.sidebar.subheader("SF12Q")
    sf12q_options = ['Q1', 'Q2', 'Q3', 'Q4']
    selected_sf12q = st.sidebar.radio("Select SF12Q", sf12q_options)

    st.sidebar.subheader("SF12M")
    sf12m_options = ['Q1', 'Q2', 'Q3', 'Q4']
    selected_sf12m = st.sidebar.radio("Select SF12M", sf12m_options)


    st.sidebar.subheader("EDAD24")
    edad24_options = ['0', '1']
    selected_edad24 = st.sidebar.radio("Select EDAD24", edad24_options)


    st.sidebar.subheader("CDH20")
    cdh20_options = ['A/A', 'A/G', 'G/G']
    selected_cdh20 = st.sidebar.radio("Select CDH20", cdh20_options)


    st.sidebar.subheader("PRKCA")
    prkca_options = ['C/C', 'C/T', 'T/T']
    selected_prkca = st.sidebar.radio("Select PRKCA", prkca_options)


    st.sidebar.subheader("TCF4")
    tcf4_options = ['A/A', 'A/T', 'T/T']
    selected_tcf4 = st.sidebar.radio("Select TCF4", tcf4_options)


    # Create a dictionary to hold the input data based on one-hot encoded columns
    input_data_dict = {}

    # Add LTE12
    input_data_dict['LTE12'] = lte12

    # Add one-hot encoded features based on selections
    for option in sf12q_options:
        input_data_dict[f'SF12Q_{option}'] = (selected_sf12q == option)

    for option in sf12m_options:
        input_data_dict[f'SF12M_{option}'] = (selected_sf12m == option)

    # GENERO_0 and GENERO_1 are fixed for this female-specific app
    input_data_dict['GENERO_0'] = False
    input_data_dict['GENERO_1'] = True

    for option in edad24_options:
         input_data_dict[f'EDAD24_{option}'] = (selected_edad24 == option)

    for option in cdh20_options:
         input_data_dict[f'CDH20_{option}'] = (selected_cdh20 == option)

    for option in prkca_options:
         input_data_dict[f'PRKCA_{option}'] = (selected_prkca == option)

    for option in tcf4_options:
         input_data_dict[f'TCF4_{option}'] = (selected_tcf4 == option)


    # Convert boolean values to integers (0 or 1)
    input_data_processed = {key: int(value) if isinstance(value, bool) else value for key, value in input_data_dict.items()}

    # Create DataFrame from processed input data, ensuring correct column order
    # Get column order from the loaded model's expected features if available,
    # otherwise use the predefined list.
    if hasattr(model, 'feature_names_in_'):
        expected_columns = model.feature_names_in_
    else:
         # Fallback to the predefined list if feature_names_in_ is not available
         expected_columns = ['LTE12', 'SF12Q_Q1', 'SF12Q_Q2', 'SF12Q_Q3', 'SF12Q_Q4', 'SF12M_Q1', 'SF12M_Q2',
                             'SF12M_Q3', 'SF12M_Q4', 'GENERO_0', 'GENERO_1', 'EDAD24_0', 'EDAD24_1', 'CDH20_A/A',
                             'CDH20_A/G', 'CDH20_G/G', 'PRKCA_C/C', 'PRKCA_C/T', 'PRKCA_T/T', 'TCF4_A/A', 'TCF4_A/T',
                             'TCF4_T/T']


    input_df = pd.DataFrame([input_data_processed], columns=expected_columns)


    # Display the input data (optional)
    # st.subheader("Input Data")
    # st.write(input_df)

    # Add a button to trigger prediction
    if st.sidebar.button("Predict ANXPHENOTYPE"):
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(input_df)[:, 1] # Probability of the positive class (1)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("The predicted ANXPHENOTYPE is 1.")
        else:
            st.error("The predicted ANXPHENOTYPE is 0.")

        if prediction_proba is not None:
            st.write(f"Probability of ANXPHENOTYPE 1: {prediction_proba[0]:.4f}")