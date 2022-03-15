#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import streamlit as st

#set the environment
import os
os.getcwd()

df = pd.read_csv("CleanedAirlineData.csv")
#df.head()


# In[14]:


# import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # train and test
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


# In[7]:


target= df.satisfaction.values


# In[9]:


df.drop(['satisfaction', 'Customer Type', 'Type of Travel', 'Flight Distance'], axis=1, inplace=True)
#Normalize Delay
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

dep_max = df['Departure Delay in Minutes'].max()
arr_max = df['Arrival Delay in Minutes'].max()
df['Departure Delay in Minutes'] = norm_func(df['Departure Delay in Minutes'])
df['Arrival Delay in Minutes'] = norm_func(df['Arrival Delay in Minutes'])
# In[11]:


model = LogisticRegression()
df_train, df_test, target_train, target_test = train_test_split(df, target, test_size=.2, random_state=42)
model.fit(df_train, target_train)


# In[12]:


prediction= model.predict(df_test)


# In[18]:


acc= accuracy_score(target_test, prediction)

# In[ ]:


# Streamlit
st.write("""
# Airline Customer Satisfaction
### Using Multiple Logistic Regression
##### By: Candy Awuor, Ran Wei, Sumaya Alzuhairy, Sunmin Ku, Yashab Narang
Customer Satisfaction is interesting when it comes to Airlines. There are many factors from seat comfort to if the
plane was late. This app predicts the probability of a customer's satisfaction after flight using some the factors as
inputs.""")

with st.form("my_form"):
    st.write("Satisfaction Predictor")
    gender_str = st.selectbox(
        'Gender',
        ('Male', 'Female'))
    age_val = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()))
    class_str = st.selectbox(
        'Flight Class',
        ('Eco', 'Eco Plus', 'Business'))
    seat_comfort = st.slider('Seat Comfort', 0, 5, 5)
    time_conv = st.slider('Departure/Arrival time convenient', 0, 5, 5)
    food_bev = st.slider('Food and Beverages', 0, 5, 5)
    gate_loc = st.slider('Gate location', 0, 5, 5)
    flight_wifi = st.slider('Inflight wifi service', 0, 5, 5)
    flight_tv = st.slider('Inflight entertainment', 0, 5, 5)
    online_support = st.slider('Online support', 0, 5, 5)
    booking_ease = st.slider('Ease of Online booking', 0, 5, 5)
    flight_service = st.slider('On-board service', 0, 5, 5)
    leg_room = st.slider('Leg room service', 0, 5, 5)
    baggage_handling = st.slider('Baggage handling', 0, 5, 5)
    checkin_conv = st.slider('Checkin service', 0, 5, 5)
    cleanliness = st.slider('Cleanliness', 0, 5, 5)
    online_boarding = st.slider('Online boarding', 0, 5, 5)
    dep_delay = st.slider('Departure Delay', 0, int(dep_max), 0)
    arr_delay = st.slider('Arrival Delay', 0, int(arr_max), 0)


    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        if gender_str == "Female":
            gender_int = 0
        elif gender_str == "Male":
            gender_int = 1

        if class_str == "Eco":
            class_int = 1
        elif class_str == "Eco Plus":
            class_int = 2
        elif class_str == "Business":
            class_int = 3


        prediction= model.predict([[age_val, class_int, seat_comfort, time_conv, food_bev, gate_loc, flight_wifi, flight_tv,
                              online_support, booking_ease, flight_service, leg_room, baggage_handling, checkin_conv,
                              cleanliness, online_boarding, dep_delay/dep_max, arr_delay/arr_max, gender_int]])[0]
        if prediction == 0:
            st.write("Customer is Dissatisfied")
        elif prediction == 1:
            st.write("Customer is Satisfied")
        else:
            st.write("Error in Calculation of Prediction")


# In[ ]:
