"""
naiveBayes
Purpose: Contains everything used to train and find accuracy for the naive bayes classifier
"""

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r"C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\keplerFluxData.csv")

# Replace dispostiion with binary label
dataset["koi_disposition"].replace(['FALSE POSITIVE', 'CONFIRMED'], [0, 1], inplace=True)

# Assigning labels and features
datasetY = dataset['koi_disposition']
datasetX = dataset[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]

# Splitting dataset
xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, test_size=.2, random_state=0)
model = BernoulliNB()

# Training model
model.fit(xTrain, yTrain)


def naiveBayesAccuracy():
    """
    naiveBayesAccuracy: this function is used to test and determine the accuracy of the naive
    bayes classifier
        Parameters:
        Returns:
            void
    """

    yPred = model.predict(xTest)
    accuracy = accuracy_score(yPred, yTest)
    accuracy = str(accuracy)
    print("Naive Bayes accuracy: " + accuracy)


def naiveBayesPrediction(curve):
    """
     naiveBayesPrediction: this function is used to make a prediction based on the trained model
         Parameters:
             curve, pandas dataframe of light curve
         Returns:
             void
     """

    datasetPredict = curve[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]
    prediction = model.predict(datasetPredict)

    print("Naive Bayes Prediction")
    if prediction == 0:
        print("No celestial body predicted")
        print()
    else:
        print("Celestial body predicted")
        print()
