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

    # Load data and save it as a csv file
    load_data()
    # Combine all the csv files into one csv file
    combine_data()
    # Clean the data
    clean_data()

# Call the main function
if __name__ == '__main__':
    main()