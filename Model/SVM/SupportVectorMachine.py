"""
SupportVectorMachine
Purpose: Contains everything used to train and find accuracy for the SVM classifier
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r"C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\keplerFluxData.csv")

# Replace disposition with binary label
dataset["koi_disposition"].replace(['FALSE POSITIVE', 'CONFIRMED'], [0, 1], inplace=True)

# Assigning labels and features
datasetY = dataset['koi_disposition']
datasetX = dataset[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]

# Splitting dataset
xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, test_size=.2, random_state=0)

# Training model
model = SVC(kernel='rbf')
model.fit(xTrain, yTrain)


def SVMAccuracy():
    """
    SVMAccuracy: this function is used to test and determine the accuracy of the SVM classifier
        Parameters:
        Returns:
            void
    """

    yPred = model.predict(xTest)
    accuracy = accuracy_score(yPred, yTest)
    accuracy = str(accuracy)
    print("Support Vector Machine accuracy: " + accuracy)


def SVMPrediction(curve):
    """
    SVMPrediction: this function is used to make a prediction based on the trained model
        Parameters:
            curve, pandas dataframe of light curve
        Returns:
            void
    """

    datasetPredict = curve[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]
    prediction = model.predict(datasetPredict)

    print('Support Vector Machine Prediction')
    if prediction == 0:
        print("No celestial body predicted")
        print()
    else:
        print("Celestial body predicted")
        print()
