# HUMAN ACTIVITY CLASSIFICATION
## Slides 
https://drive.google.com/file/d/10wdNs44kHGGQ94X5zTZvtYB06YTe1lmD/view?usp=sharing

### Usage and File Structure ?
1. Download the PAMAP2_Dataset from https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
2. Download the PAMAP2_data.csv into root/src directory from the following OneDrive Link : https://iiitaphyd-my.sharepoint.com/:f:/g/personal/divyansh_t_research_iiit_ac_in/EhHLy69jD4xHqiqZc_FZlrIBl9JDPdDgHOvXD7y1HzbF6A?e=YOjACc
   (Data required in a particular format for running Project_Trees.ipynb and Project_NN.ipynb)
3. Make sure to place the PAMAP2_data.csv in the root/src directory (in the same directory as the Project_Trees.ipynb and Project_NN.ipynb)
4. Run the 'main.py' file in the root directory using the following command:
   ```python
   python main.py
   ```
   which preprocesses the data

5. The nootebooks for Logistic Regression, Decision Trees, SVM, Neural Network are in src directory and can be run directly

### Description
1. The research paper on Human Activity Classification is an exploratory paper that explores different classification models which run over the IMU sensor data of the participants. 
2. This area of research is witnessing a wide range of applications such as in smart wearables which gathers data from various sensors and classifies the activity that is being performed.  
3. There were a total of 12 classes and the data was in the form of timed sensor readings yielding 40 metrics . 
