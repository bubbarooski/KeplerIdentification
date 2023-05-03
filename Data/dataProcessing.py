"""
dataProcessing
Purpose: This is used to find values related to the curves for the classical classifiers (KNN,
NB, and SVM). You specify a max and min index of the main Kepler dataset to iterate through
and it will calculate max, min, mean, max-min, and variance of flux for each of the indices
and export them to a dataset called "keplerFluxData.csv"
"""

import pandas as pd
from Data.retreiveImageData import processCurve

kic = "KIC "
minIndex = int(input("Min index: "))
maxIndex = int(input("Max index: "))
currentIndex = minIndex

# Initializing new columns in dataframe
dataset = pd.read_csv("keplerDataset.csv")
dataset['maxFlux'] = float(0)
dataset['minFlux'] = float(0)
dataset['meanFlux'] = float(0)
dataset['p2pFlux'] = float(0)
dataset['varianceFlux'] = float(0)

for row in range(maxIndex - minIndex - 1):
    kepID = dataset['kepid'][currentIndex]
    kepID = str(kepID)
    kepID = kic + kepID
    curve = processCurve(kepID)

    # Determining values
    maxFlux = curve['flux'].max()
    minFlux = curve['flux'].min()
    meanFlux = curve['flux'].mean()
    p2pFlux = maxFlux - minFlux
    varianceFlux = curve['flux'].var()

    # Putting values in dataframe
    dataset.at[currentIndex, "maxFlux"] = maxFlux
    dataset.at[currentIndex, "minFlux"] = minFlux
    dataset.at[currentIndex, "meanFlux"] = meanFlux
    dataset.at[currentIndex, "p2pFlux"] = p2pFlux
    dataset.at[currentIndex, "varianceFlux"] = varianceFlux

    currentIndex += 1
    print("Finished index " + str(currentIndex))

dataset.to_csv("keplerFluxData.csv")
print(dataset.head(25))
