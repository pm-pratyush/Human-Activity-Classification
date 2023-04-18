# Import all the required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from DataLoader import *
from DataCleaner import *

# Main function
def main():
    # Create a folder to store the models
    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    # Create a folder to store the results
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Create a folder to store the data in CSV format
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if (len(os.listdir(data_folder)) == 4) and (all([file.startswith('t') and file.endswith('.csv') for file in os.listdir(data_folder)])):
        pass
    else:
        # Step 1: Convert the data to CSV format
        convert_data_to_csv(data_folder)
        print("Data converted to CSV format successfully !!!")

        # Step 2: Load the data into a pandas dataframe
        dataCollection = load_data(data_folder)
        print("Data loaded successfully into a pandas dataframe !!!")

        # Step 3: Clean the data
        dataCollection = clean_data(dataCollection)
        print("Data cleaned successfully !!!")

        # Step 4: Split the data into train and test
        train_X, train_Y, test_X, test_Y = split_data(dataCollection)
        print("Data split into train and test successfully !!!")
        print("Train data shape: ", train_X.shape)
        print("Test data shape: ", test_X.shape)

        # Step 5: Write the train and test data to CSV files
        write_data_to_csv(train_X, train_Y, test_X, test_Y, data_folder)
        print("Train and test data written to CSV files successfully !!!")

    train_X = pd.read_csv(data_folder + '/' + 'train_X.csv')
    train_Y = pd.read_csv(data_folder + '/' + 'train_y.csv')
    test_X = pd.read_csv(data_folder + '/' + 'test_X.csv')
    test_Y = pd.read_csv(data_folder + '/' + 'test_y.csv')
    print("Train and test data loaded from CSV files successfully !!!")

# Call the main function
if __name__ == '__main__':
    main()


