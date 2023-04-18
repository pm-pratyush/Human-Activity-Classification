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

    # Step 1: Convert the data to CSV format
    convert_data_to_csv(data_folder)

    # Step 2: Load the data into a pandas dataframe
    dataCollection = load_data(data_folder)
    print("Data loaded successfully into a pandas dataframe !!!")
    print(dataCollection.head())
    
    # Step 3: Clean the data

# Call the main function
if __name__ == '__main__':
    main()