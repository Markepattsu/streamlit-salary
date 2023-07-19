import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.datasets import make_regression
import lightgbm as lgb

import os


st.write("""
# Salary Prediction app

""")

st.sidebar.header('User Input Features')


#collect user input features into a dataframe

def user_input_features():

    # data = {}
    
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
        data[f'New_Age_Range_{age_range}'] = int(age_range == selected_age_range)

    for industry in industries:
        data[f'What_industry_do_you_work_in__{industry}'] = int(industry == selected_industry)

    for continent in continents:
        data[f'Continent_{continent}'] = int(continent == selected_continent)

    for exp in experience_overall:
        data[f'How_many_years_of_professional_work_experience_do_you_have_overall__{exp}'] = int(exp == selected_experience_overall)
    
    for exp in experience_field:
        data[f'How_many_years_of_professional_work_experience_do_you_have_in_your_field__{exp}'] = int(exp == selected_experience_field)

    for degree in degrees:
        data[f'What_is_your_highest_level_of_education_completed__{degree}'] = int(degree == selected_degree)

    for gender in genders:
        data[f'What_is_your_gender__{gender}'] = int(gender == selected_gender)

    # data[f'New_Age_Range_{age_range}'] = age_range == selected_age_range
    # data[f'What_industry_do_you_work_in__{industry}'] = industry == selected_industries
    # data[f'Continent_{continent}'] = continent == selected_continents
    # data[f'How_many_years_of_professional_work_experience_do_you_have_overall__{exp}'] = exp == selected_experience_overall
    # data[f'How_many_years_of_professional_work_experience_do_you_have_in_your_field__{exp}'] = exp == selected_experience_field
    # data[f'What_is_your_highest_level_of_education_completed__{degree}'] = degree == selected_degree
    # data[f'What_is_your_gender__{gender}'] = gender == selected_gender
        
    features = pd.DataFrame(data, index=[0])
    return features

# def one_hot_encode(df):
#     # Use pandas get_dummies to one-hot encode the data
#     encoded_df = pd.get_dummies(df)
#     return encoded_df

df = user_input_features()

# encoded_df = one_hot_encode(df)


#     data = {}
#     data['New Age Range'] = selected_age_range
#     data['What industry do you work in?'] = selected_industries
#     data['Continent'] = selected_continents
#     data['How many years of professional work experience do you have overall?'] = selected_experience_overall
#     data['How many years of professional work experience do you have in your field?'] = selected_experience_field
#     data['What is your highest level of education completed?'] = selected_degree
#     data['What is your gender?'] = selected_gender

#     # Perform one-hot encoding
#     # encoded_data = {}
#     # for key, value in data.items():
#     #     if isinstance(value, list):
#     #         for item in value:
#     #             encoded_data[f'{key}__{item}'] = int(item in value)
#     #     else:
#     #         encoded_data[key] = value

#     features = pd.DataFrame(encoded_data, index=[0])
#     return features

# df = user_input_features()



# encode = ['New Age Range', 'What industry do you work in?', 'Continent', 'How many years of professional work experience do you have overall?', 'How many years of professional work experience do you have in your field?', 'What is your highest level of education completed?', 'What is your gender?']

# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df, dummy], axis=1)
#     del df[col]

df = df.iloc[:1]



# Displays the user input features

st.subheader('User Input Features')

model = pickle.load(open('model_lgbm.pkl','rb'))

predictions = model.predict(df)
# Display the predictions
st.subheader('Predicted')
st.write(predictions)




