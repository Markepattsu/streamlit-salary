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

    # data = {'New_Age_Range': age_ranges,
    #        'What_industry_do_you_work_in': industries,
    #        'Continent': continents,
    #        'How_many_years_of_professional_work_experience_do_you_have_overall': experience_overall,
    #        'How_many_years_of_professional_work_experience_do_you_have_in_your_field': experience_field,
    #        'What_is_your_highest_level_of_education_completed': degrees,
    #        'What_is_your_gender': genders}

    # data['Age Range'] = selected_age_range
    # data['Industries'] = selected_industries
    # data['Continents'] = selected_continents
    # data['Overall Professional Work Experience'] = selected_experience_overall
    # data['Professional Work Experience in the Field'] = selected_experience_field
    # data['Highest Level of Education Completed'] = selected_degree
    # data['Gender'] = selected_gender
                                          
    # for age_range in age_ranges:
    #     data[f'New_Age_Range_{age_range}'] = age_range in selected_age_range

    # for industry in industries:
    #     data[f'What_industry_do_you_work_in__{industry}'] = industry in selected_industries

    # for continent in continents:
    #     data[f'Continent_{continent}'] = continent in selected_continents

    # for exp in experience_overall:
    #     data[f'How_many_years_of_professional_work_experience_do_you_have_overall__{exp}'] = exp in selected_experience_overall
    
    # for exp in experience_field:
    #     data[f'How_many_years_of_professional_work_experience_do_you_have_in_your_field__{exp}'] = exp in selected_experience_field

    # for degree in degrees:
    #     data[f'What_is_your_highest_level_of_education_completed__{degree}'] = degree in selected_degree

    # for gender in genders:
    #     data[f'What_is_your_gender__{gender}'] = gender in selected_gender

    # data = {
    #     'New_Age_Range': [age_range == selected_age_range for age_range in age_ranges],
    #     'What_industry_do_you_work_in': [industry in selected_industries for industry in industries],
    #     'Continent': [continent in selected_continents for continent in continents],
    #     'How_many_years_of_professional_work_experience_do_you_have_overall': [exp == selected_experience_overall for exp in experience_overall],
    #     'How_many_years_of_professional_work_experience_do_you_have_in_your_field': [exp == selected_experience_field for exp in experience_field],
    #     'What_is_your_highest_level_of_education_completed': [degree in selected_degree for degree in degrees],
    #     'What_is_your_gender': [gender in selected_gender for gender in genders]
    # }

    # data = {
    #     'New Age Range': [age_range == selected_age_range for age_range in age_ranges],
    #     'What industry do you work in?': [industry in selected_industries for industry in industries],
    #     'Continent': [continent in selected_continents for continent in continents],
    #     'How many years of professional work experience do you have overall?': [exp == selected_experience_overall for exp in experience_overall],
    #     'How many years of professional work experience do you have in your field?': [exp == selected_experience_field for exp in experience_field],
    #     'What is your highest level of education completed?': [degree == selected_degree for degree in degrees],
    #     'What is your gender?': [gender == selected_gender for gender in genders]
    # }

    data = {}
    data['New Age Range'] = [selected_age_range]
    data['What industry do you work in?'] = [selected_industries]
    data['Continent'] = [selected_continents]
    data['How many years of professional work experience do you have overall?'] = [selected_experience_overall]
    data['How many years of professional work experience do you have in your field?'] = [selected_experience_field]
    data['What is your highest level of education completed?'] = [selected_degree]
    data['What is your gender?'] = [selected_gender]

    features = pd.DataFrame(data)
    return features

df = user_input_features()

features_df = pd.get_dummies(df, columns=['New Age Range', 'What industry do you work in?','Continent','How many years of professional work experience do you have overall?',
                                         'How many years of professional work experience do you have in your field?','What is your highest level of education completed?','What is your gender?'])
    
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

predictions = model.predict(features_df)
# Display the predictions
st.subheader('Predicted')
st.write(predictions)




