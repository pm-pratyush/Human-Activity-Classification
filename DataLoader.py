# Import all the required libraries
import os
import tqdm
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a list of files from which data is to be loaded
list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                 'PAMAP2_Dataset/Protocol/subject102.dat',
                 'PAMAP2_Dataset/Protocol/subject103.dat',
                 'PAMAP2_Dataset/Protocol/subject104.dat',
                 'PAMAP2_Dataset/Protocol/subject105.dat',
                 'PAMAP2_Dataset/Protocol/subject106.dat',
                 'PAMAP2_Dataset/Protocol/subject107.dat',
                 'PAMAP2_Dataset/Protocol/subject108.dat',
                 'PAMAP2_Dataset/Protocol/subject109.dat']

# Create a dictionary to store the activity ID and its corresponding activity
activityIDdict = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

# Create a list of column names: Total 54 columns
colNames = ["timestamp", "activityID","heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

columns = colNames + IMUhand + IMUchest + IMUankle

# Create a function to load data and store in a single csv file
def convert_data_to_csv(folder):
    if len(os.listdir(folder)) == 0:
        print('\nConverting data to csv files...')
        # Store the data in a different csv file for each subject
        for file in tqdm.tqdm(list_of_files):
            # Read the data from the file
            data = pd.read_csv(file, sep = ' ', header = None)
            # Assign the column names
            data.columns = columns
            data.to_csv(folder + '/' + file.split('/')[-1].split('.')[0] + '.csv', index = False)

        print('Data converted to csv files successfully!!!')
        delete_folder()
    else:
        print('Data Folder already exists!!!')
        delete_folder()

def delete_folder(folder = 'PAMAP2_Dataset'):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print('\n' + folder + ' folder deleted successfully!!!')

def load_data(folder = 'data'):
    # Load all data from all csv files
    print('\nLoading data...')

    data = pd.DataFrame()
    for file in tqdm.tqdm(os.listdir(folder)):
        data = data.append(pd.read_csv(folder + '/' + file))
    return data
