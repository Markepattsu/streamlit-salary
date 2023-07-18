import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.datasets import make_regression
import os



st.write("""
# Salary Prediction app

""")

st.sidebar.header('User Input Features')


#collect user input features into a dataframe

def user_input_features():
    
    data = {}

    age_ranges = ['Under 18', '18-34', '35-54', '55 or over']
    selected_age_range = st.sidebar.selectbox('Age Range', age_ranges)
    
    industries = [
        'Accounting/Banking/Finance',
        'Agriculture or Forestry',
        'Art Design',
        'Business or Consulting',
        'Computing or Tech',
        'Education Higher Education',
        'Education Primary Secondary',
        'Engineering or Manufacturing',
        'Entertainment',
        'Government and Public Administration',
        'Health care',
        'Hospitality & Events',
        'Insurance',
        'Law',
        'Law Enforcement & Security',
        'Leisure, Sport & Tourism',
        'Marketing, Advertising & PR',
        'Media & Digital',
        'Nonprofits',
        'Property or Construction',
        'Recruitment or HR',
        'Retail',
        'Sales',
        'Social Work',
        'Transport or Logistics',
        'Utilities & Telecommunications',
    ]       
    
    selected_industries = st.sidebar.selectbox('Industries', industries)
    continents = [
        'Africa',
        'Asia',
        'Europe',
        'North America',
        'Oceania',
        'South America',
    ]
    selected_continents = st.sidebar.selectbox('Continents', continents)
    
    experience_overall = [
        '1 year or less',
        '2-4 years',
        '5-7 years',
        '8-10 years',
        '11-20 years',
        '21-30 years',
        '31-40 years',
        '41 years or more',
    ]
    selected_experience_overall = st.sidebar.selectbox('Overall Professional Work Experience', experience_overall)
    
    experience_field = [
        '1 year or less',
        '2-4 years',
        '5-7 years',
        '8-10 years',
        '11-20 years',
        '21-30 years',
        '31-40 years',
        '41 years or more',
    ]
    selected_experience_field = st.sidebar.selectbox('Professional Work Experience in the Field', experience_field)
    
    degrees = [
        'College degree',
        'High School',
        'Master\'s degree',
        'PhD',
        'Professional degree (MD, JD, etc.)',
        'Some college',
    ]
    selected_degree = st.sidebar.radio('Highest Level of Education Completed', degrees)

    genders = [
        'Man',
        'Non-binary',
        'Other or prefer not to answer',
        'Prefer not to answer',
        'Woman',
    ]
    selected_gender = st.sidebar.radio('Gender', genders)
                                                     
    for age_range in age_ranges:
        data[f'New_Age_Range_{age_range}'] = age_range == selected_age_range

    for industry in industries:
        data[f'What_industry_do_you_work_in__{industry}'] = industry in selected_industries

    for continent in continents:
        data[f'Continent_{continent}'] = continent in selected_continents

    for exp in experience_overall:
        data[f'How_many_years_of_professional_work_experience_do_you_have_overall__{exp}'] = exp == selected_experience_overall
    
    for exp in experience_field:
        data[f'How_many_years_of_professional_work_experience_do_you_have_in_your_field__{exp}'] = exp == selected_experience_field

    for degree in degrees:
        data[f'What_is_your_highest_level_of_education_completed__{degree}'] = degree in selected_degree



    for gender in genders:
        data[f'What_is_your_gender__{gender}'] = gender in selected_gender


    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


encode = ['Age Range', 'Industries','Continents','Overall Professional Work Experience','Professional Work Experience in the Field','Highest Level of Education Completed','Gender']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# select only the first row (the user input data)

df = df.iloc[:1]


# Displays the user input features

st.subheader('User Input Features')

model_path = 'C:/Users/tanak/OneDrive/Desktop/Desk/Work/Yr 2/Sem 1/MLDP/Proj/model_lgbm.pkl'

if os.path.exists(model_path):
    # Load the model
    model = joblib.load(model_path)
    # Rest of your Streamlit app code
else:
    st.error('Model file not found. Please verify the file path.')




model = joblib.load(r'C:\Users\tanak\OneDrive\Desktop\Desk\Work\Yr 2\Sem 1\MLDP\Proj\model_lgbm.pkl')


if os.path.exists(model_path):
    # Load the model
    model = joblib.load(model_path)
    # Make predictions using the loaded model
    predictions = model.predict(df)
    prediction_proba = model.predict_proba(df)
    # Display the predictions
    st.subheader('Predicted')
    st.write(predictions)
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
else:
    st.error('Model file not found. Please verify the file path.')




st.subheader('Prediction Probability')
st.write(prediction_proba) 

