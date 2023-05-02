import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r"C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\keplerFluxData.csv")

dataset["koi_disposition"].replace(['FALSE POSITIVE', 'CONFIRMED'], [0, 1], inplace=True)

datasetY = dataset['koi_disposition']
datasetX = dataset[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]

xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, test_size=.2, random_state=0)

model = KNeighborsClassifier()
model.fit(xTrain, yTrain)


def knnAccuracy():
    yPred = model.predict(xTest)
    accuracy = accuracy_score(yPred, yTest)
    accuracy = str(accuracy)
    print("KNN accuracy: " + accuracy)


def knnPrediction(curve):
    datasetPredict = curve[['maxFlux', 'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux']]
    prediction = model.predict(datasetPredict)

    print("KNN Prediction")
    if prediction == 0:
        print("No celestial body predicted")
        print()
    else:
        print("Celestial body predicted")
        print()
