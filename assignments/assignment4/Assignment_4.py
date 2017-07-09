
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[1]:

# Make sure that you uncomment all your plotting stuff before submit your assignment!!!

import pandas as pd
import numpy as np

# Function used to calculate the difference between the violation issue date and the hearing date
def diff_seconds(date2, date1):
    # this is a function that I found on coursera
    res =  (date2.dt.date - date1.dt.date).dt.total_seconds()
    res += (date2.dt.hour - date1.dt.hour) * 3600
    res += (date2.dt.minute - date1.dt.minute) * 60
    res += date2.dt.second - date1.dt.second
    return res

# Function to add location data
def add_location(data):
    # addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates.
    addresses = pd.read_csv('addresses.csv', index_col='ticket_id')
    # Read the GPS coordinates for each address.
    lat_lons = pd.read_csv('latlons.csv')
    # Let's start merging everything.
    data = data.merge(addresses, left_index=True, right_index=True, how='left')
    # Reset index before merge. set it again after.
    data = data.reset_index().merge(lat_lons, on='address', how='left').set_index('ticket_id')
    return data

# This is where you read the CSV's and parse all the data that you need
def read_csv_and_preprocess(file_name):
    # Let's read in our training dataset
    data_train = pd.read_csv(file_name,encoding = "ISO-8859-1", index_col='ticket_id')
    
    # Only do this on the training data. Test data does not have compliance info
    if file_name == 'train.csv':
        # Drop rows (violation tickets) that have Nan's for targets. They are not used during evaluation
        data_train = data_train[data_train.compliance.notnull()]

        # Figure out the percentage of violations that were paid on time.
        compliance_counts = data_train.compliance.value_counts()
        print('Percentage of people that paid on-time in training dataset: {}%'.format(str(100*compliance_counts[1]/compliance_counts[0])))
    
    # Convert columns to date-time. Ex: 2005-03-21 10:30:00
    # Start by filling NaNs with some value
    data_train.ticket_issued_date = pd.to_datetime(data_train.ticket_issued_date.fillna('1900-01-01 00:00:00'))
    data_train.hearing_date = pd.to_datetime(data_train.hearing_date.fillna('1900-01-01 00:00:00'))
    # Now convert to date-time data type.
    data_train.ticket_issued_date =  pd.to_datetime(data_train.ticket_issued_date, format='%Y-%m-%d %H:%M:%S')
    data_train.hearing_date =  pd.to_datetime(data_train.hearing_date, format='%Y-%m-%d %H:%M:%S')
    
    ######################
    # feature engineering
    #####################
    # Start by computing the difference betweent the ticket issue date and the hearing date.
    data_train['issue_hearing_date_diff'] = diff_seconds(data_train.hearing_date,data_train.ticket_issued_date)
    # Now let's add logitude and latitude
    # addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates.
    data_train = add_location(data_train)
    # Let's add datetime info to the dataset
    data_train['issue_day'] = data_train.ticket_issued_date.dt.day
    data_train['issue_month'] = data_train.ticket_issued_date.dt.month
    data_train['issue_year'] = data_train.ticket_issued_date.dt.year
    data_train['issue_weekday'] = data_train.ticket_issued_date.dt.weekday
    
    return data_train # return your data-frame

# Get your X data
def get_X(data):
    columns_to_keep = ['issue_hearing_date_diff','fine_amount','admin_fee','state_fee','late_fee', 'discount_amount',
                       'judgment_amount','lat','lon','issue_day','issue_month','issue_year','issue_weekday']
    # Return. Make sure you replace NaNs with zeros. Only lat and lon columns have NaNs
    return data.loc[:, columns_to_keep].fillna(0)

def get_y(data):
    # Return the target variable as a pandas series. This one is super easy
    return data.compliance

from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt
def blight_model():
    
    # Your code here
    # Let's read your data
    data_train = read_csv_and_preprocess('train.csv')
    
    # Get your input data for training
    X = get_X(data_train)
    # Get your target variable
    y = get_y(data_train)
    
    # Declare your scaler object.
    scaler = MinMaxScaler()
    # Split your training and test data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    # Scale all of your data.
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #######################
    # Now, Declare a model.
    #######################
    # Didn't try this one
    #nbclf = GaussianNB().fit(X_train_scaled, y_train)
    # This one got me to pass
    model = LogisticRegression().fit(X_train_scaled, y_train)
    # Neural net doesn't work. doesn't have descicion function
#     model = MLPClassifier(hidden_layer_sizes = [20, 20], solver='lbfgs',
#                      random_state = 0).fit(X_train, y_train)
    # This one takes way too long
    #model = SVC(kernel='rbf').fit(X_train_scaled, y_train)

    # Let's print out the accuracy of our model just as a reference.
    print('Accuracy of training set: {:.2f}'.format(model.score(X_train_scaled, y_train)))
    print('Accuracy of test set: {:.2f}'.format(model.score(X_test_scaled, y_test)))

    # Get variables to maek ROC
    y_score_model = model.decision_function(X_test_scaled)
    fpr_model, tpr_model, _ = roc_curve(y_test, y_score_model)
    roc_auc_model = auc(fpr_model, tpr_model)

    # Plot your ROC and print AOC score
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_model, tpr_model, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_model))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (Violation paid on-time classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
    ###################################################
    # let's got the decision score for the test dataset
    ###################################################
    # Ready CSV file
    data_test = read_csv_and_preprocess('test.csv')
    # Get only the columns that you are going to use
    X_val = get_X(data_test)
    # Scale your data
    X_val_scaled = scaler.transform(X_val)
    # Get the confidence scores for each violation
    X_val_score = model.decision_function(X_val_scaled)

    # now return your confidence score but convert it to a pandas data frame
    return pd.Series(data=X_val_score, index=X_val.reset_index()['ticket_id'], dtype="float32")

blight_model().head()
                 


# In[ ]:



