"""
Driver
Purpose: This is the driver for the main program and allows the user to view the accuracy of all models or
enter an index in the Kepler database to view its light curve, the predictions of all models, and the actual
classification.
"""

from Model.NB.naiveBayes import naiveBayesAccuracy, naiveBayesPrediction
from Model.KNN.kNearestNeighbors import knnAccuracy, knnPrediction
from Model.SVM.SupportVectorMachine import SVMAccuracy, SVMPrediction
from Model.tfCNN.tfCNNtest import cnnAccuracy, cnnTest
from Model.tfGRU.tfGRUtest import gruAccuracy, gruTest
from Model.tfMLP.tfMLPtest import mlpAccuracy, mlpTest
import Data.singleLightCurve as single
from Data.retreiveImageData import processCurve, plotCurvePretty, plotCurve
import pandas as pd

menu = 0
keplerDataset = dataset = pd.read_csv(r"C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\keplerDataset.csv")
kic = "KIC "
png = ".png"

while menu != -1:
    print()
    print("Enter 1 for accuracy")
    print("Enter 2 for specific curve")
    print("Enter -1 to exit")
    menu = int(input("Enter choice: "))

    if menu == 1:
        naiveBayesAccuracy()
        knnAccuracy()
        SVMAccuracy()
        cnnAccuracy()
        gruAccuracy()
        mlpAccuracy()

    if menu == 2:
        index = int(input("Enter index from 0-999: "))
        if index < 0 or index > 999:
            print("Invalid index ")
        else:
            curve = single.processSingleLightcurve(index)
            starID = curve['kepid'][0]
            status = curve['koi_disposition'][0]
            plotCurvePretty(processCurve(starID))

            naiveBayesPrediction(curve)
            knnPrediction(curve)
            SVMPrediction(curve)

            status = keplerDataset['koi_disposition'][index]
            if status == "CONFIRMED":
                confirmed = 1
            else:
                confirmed = 0

            curve = processCurve(starID)
            plotCurve(curve, starID, confirmed)

            kepID = dataset['kepid'][index]
            kepID = str(kepID)
            kepID = kic + kepID
            kepID += png

            cnnTest(kepID, confirmed)
            gruTest(kepID, confirmed)
            mlpTest(kepID, confirmed)

            print("Actual status: ")
            if status == "CONFIRMED":
                print("Celestial body found")
            else:
                print("No celestial body found")

