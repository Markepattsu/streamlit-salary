import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.externals
import os

st.image('https://www.zdnet.com/a/img/resize/77fc992324bb5da825fa58eca0be48a2e8d3a146/2020/07/21/8f8c5e3b-1eb7-4100-b4d8-0059c89cd8e6/istock-1213497796-2.jpg?auto=webp&width=1280', width=800)

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
    selected_industry = st.sidebar.selectbox('Industries', industries)
    
    continents = [
        'Africa',
        'Asia',
        'Europe',
        'North America',
        'Oceania',
        'South America',
    ]
    selected_continent = st.sidebar.selectbox('Continents', continents)
    
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

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

df = df.iloc[:1]

feature_df = pd.read_csv('cleared.csv')

exclude_columns = ['Salary', 'Please indicate the currency', 'What is your race? (Choose all that apply.)','What country do you work in?','How old are you?']  # Add the column names to be excluded

X = feature_df.drop(exclude_columns, axis=1)

y = feature_df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2200187)

feature_names = [
    'New Age Range_18-34',
    'New Age Range_35-54',
    'New Age Range_55 or over',
    'New Age Range_under 18',
    'What industry do you work in?_Accounting, Banking & Finance',
    'What industry do you work in?_Agriculture or Forestry',
    'What industry do you work in?_Art & Design',
    'What industry do you work in?_Business or Consulting',
    'What industry do you work in?_Computing or Tech',
    'What industry do you work in?_Education (Higher Education)',
    'What industry do you work in?_Education (Primary/Secondary)',
    'What industry do you work in?_Engineering or Manufacturing',
    'What industry do you work in?_Entertainment',
    'What industry do you work in?_Government and Public Administration',
    'What industry do you work in?_Health care',
    'What industry do you work in?_Hospitality & Events',
    'What industry do you work in?_Insurance',
    'What industry do you work in?_Law',
    'What industry do you work in?_Law Enforcement & Security',
    'What industry do you work in?_Leisure, Sport & Tourism',
    'What industry do you work in?_Marketing, Advertising & PR',
    'What industry do you work in?_Media & Digital',
    'What industry do you work in?_Nonprofits',
    'What industry do you work in?_Property or Construction',
    'What industry do you work in?_Recruitment or HR',
    'What industry do you work in?_Retail',
    'What industry do you work in?_Sales',
    'What industry do you work in?_Social Work',
    'What industry do you work in?_Transport or Logistics',
    'What industry do you work in?_Utilities & Telecommunications',
    'Continent_Africa',
    'Continent_Asia',
    'Continent_Europe',
    'Continent_North America',
    'Continent_Oceania',
    'Continent_South America',
    'How many years of professional work experience do you have overall?_1 year or less',
    'How many years of professional work experience do you have overall?_11 - 20 years',
    'How many years of professional work experience do you have overall?_2 - 4 years',
    'How many years of professional work experience do you have overall?_21 - 30 years',
    'How many years of professional work experience do you have overall?_31 - 40 years',
    'How many years of professional work experience do you have overall?_41 years or more',
    'How many years of professional work experience do you have overall?_5-7 years',
    'How many years of professional work experience do you have overall?_8 - 10 years',
    'How many years of professional work experience do you have in your field?_1 year or less',
    'How many years of professional work experience do you have in your field?_11 - 20 years',
    'How many years of professional work experience do you have in your field?_2 - 4 years',
    'How many years of professional work experience do you have in your field?_21 - 30 years',
    'How many years of professional work experience do you have in your field?_31 - 40 years',
    'How many years of professional work experience do you have in your field?_41 years or more',
    'How many years of professional work experience do you have in your field?_5-7 years',
    'How many years of professional work experience do you have in your field?_8 - 10 years',
    'What is your highest level of education completed?_College degree',
    'What is your highest level of education completed?_High School',
    'What is your highest level of education completed?_Master\'s degree',
    'What is your highest level of education completed?_PhD',
    'What is your highest level of education completed?_Professional degree (MD, JD, etc.)',
    'What is your highest level of education completed?_Some college',
    'What is your gender?_Man',
    'What is your gender?_Non-binary',
    'What is your gender?_Other or prefer not to answer',
    'What is your gender?_Prefer not to answer',
    'What is your gender?_Woman'
]

sanitized_feature_names = [re.sub(r'\W+', '_', feature) for feature in feature_names]

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 50,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Create a LightGBM Dataset with sanitized feature names
train_data = lgb.Dataset(X_train, label=y_train, feature_name=sanitized_feature_names)

# Train the LightGBM model
lgbm = lgb.train(params, train_data, num_boost_round=100)



# Displays the user input features

# model = pickle.load(open('model_lgbm.pkl','rb'))
# predictions = model.predict(df)

predictions = lgbm.predict(df)
# Display the predictions
st.subheader('Predicted Annual Salary in USD')
st.write(predictions)




