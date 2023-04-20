# Import all the required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to clean data
def clean_data(data):
    data = data.dropna()
    data = data.reset_index(drop=True)

    # Remove certain columns
    data = data.drop(['timestamp'], axis=1)
    # Remove the orientation columns
    data = data.drop(['handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4'], axis=1)
    data = data.drop(['chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4'], axis=1)
    data = data.drop(['ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'], axis=1)

    # For the heart rate, fill missing values with previous timestamp's heart rate
    data['heartRate'] = data['heartRate'].fillna(method='ffill')

    # For any other missing values, fill them with last value
    data = data.fillna(method='ffill')

    # Normalize the data
    # data = (data - data.mean()) / data.std()
    # discard data with NaN values
    data = data.dropna()
    data = data.reset_index(drop=True)

    # disacrd data with activityID = 0
    data = data[data['activityID'] != 0]
    data = data.reset_index(drop=True)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    return data

# Split the data into train and test
def split_data(data):
    # Split the data into train and test
    train = data.sample(frac=0.8, random_state=200)
    test = data.drop(train.index)

    # Split the train and test data into features and labels
    train_X = train.drop(['activityID'], axis=1)
    train_Y = train['activityID']
    test_X = test.drop(['activityID'], axis=1)
    test_Y = test['activityID']

    # normalize train and test data
    # train_X = (train_X - train_X.mean()) / train_X.std()
    # test_X = (test_X - test_X.mean()) / test_X.std()
    
    return train_X, train_Y, test_X, test_Y