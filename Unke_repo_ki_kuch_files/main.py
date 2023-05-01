# Zach Blum, Navjot Singh, Aristos Athens

"""
    Main file for running project.
"""

import datetime
print("date and time ho gya")

import discriminative_models
print("discriminative_models ho gya")
import neural_net
print("neural_net ho gya")
import util
print("util ho gya")
import decision_trees
print("decision_trees ho gya")

import makeCleanData
print("makeCleanData ho gya")
from enum_types import *

# # ------------------------------------- Main ------------------------------------- #


def main():
    """
        Create Learner object, train it, get predictions.
    """

    output_folder_name = "./../output/"
    models_folder_name = "./../models/"
    data_folder_name = './../data/'

    # makeCleanData.convert_data("./../data/")
    # raise Exception("here")

    # ---------------------------- Neural Net Model -------------------------------------
    # Create DeepLearner object, train it

    print("Hemlu before training")

    learner = neural_net.DeepLearner(data_folder_name,
                                     output_folder_name,
                                     models_folder_name,
                                     batch_size=200,
                                     architecture=ArchitectureType.MLP_multiclass
                                     )
    learner.train(epochs=2)

    print("NN model training ho gyi")

    # Plot learner info
    accuracy = learner.history.history["accuracy"]
    loss = util.normalize(learner.history.history["loss"])

    print("On line 57")

    time_string = str(datetime.datetime.now().isoformat(' ', 'minutes'))
    info = learner.info_string()

    print("On line 62")

    util.plot(data=[accuracy, loss],
              title="Accuracy, Loss v Epochs",
              x_label="Epochs",
              labels=["Training Accuracy", "Normalized Loss"],
              file_name=output_folder_name + "Accuracy, Loss v Time " + time_string + ".png",
              fig_text=info
              )
    

    print("On line 73")
    
    accuracy = learner.accuracy()

    print("On line 77")

    print(accuracy)

    # ----------------------- Logistic Regression model ---------------------------------
    learner = discriminative_models.DiscriminativeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                          percent_validation=0.3, epsilon=25.0, learning_rate=1e-2,
                                                          use_lib=True, model='svm')  # model can be 'log_reg' or 'svm'
    
    print("On line 86")

    learner.tune_hyperparamter()

    print("On line 90")

    learner.train(None)

    print("On line 94")

    learner.predict(None)

    print("On line 98")

    learner = discriminative_models.DiscriminativeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                          percent_validation=0.3, epsilon=25.0, learning_rate=1e-2,
                                                          use_lib=True, model='log_reg')  # model can be 'log_reg' or 'svm'
    
    print("On line 104")

    learner.tune_hyperparamter()

    print("On line 108")

    learner.train(None)

    print("On line 112")

    learner.predict(None)

    print("On line 116")

    # # ----------------------- Decision Trees ---------------------------------------------
    learner = decision_trees.DecisionTreeLearner(data_folder_name, output_folder_name, models_folder_name,
                                                 percent_validation=0.15)
    learner.normal_trees()
    learner.test_trees()
    learner.random_forest()
    learner.test_RF()
    learner.boosted_trees()
    learner.test_boosted()


if __name__ == "__main__":
    print("Main ko call kr rhe hai ab")
    main()
    print("Maa kasam maza aa gya")
