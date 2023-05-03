"""
datasetRetrieval
Purpose: This is used to download and export light curves to the Lightcurves directory. You
specify a max and min index of the main Kepler dataset to iterate through and it will retrieve
the light curves and export it to either the "Confirmed" or "Not Confirmed" folders.
"""

import pandas as pd
import retreiveImageData

minIndex = 0
maxIndex = 1
kic = "KIC "

minIndex = int(input("Input minimum index of downloading: "))
maxIndex = int(input("Input maximum index of downloading: "))
currentIndex = minIndex

# transformDataset.dataDriver("keplerTrainingData.csv")
dataset = pd.read_csv("keplerDataset.csv")

for x in range(maxIndex - minIndex + 1):
    kepID = dataset['kepid'][currentIndex]
    kepID = str(kepID)
    kepID = kic + kepID
    print("Exporting light curve: " + kepID + " at index " + str(currentIndex))

    # Flag to determine if light curve has confirmed body or not
    status = dataset['koi_disposition'][currentIndex]
    if status == "CONFIRMED":
        confirmed = 1
    else:
        confirmed = 0

    # Retrieving curve and saving to proper directory
    curve = retreiveImageData.processCurve(kepID)
    retreiveImageData.plotCurve(curve, kepID, confirmed)
    currentIndex += 1
