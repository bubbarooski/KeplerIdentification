import transformDataset
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

    status = dataset['koi_disposition'][currentIndex]
    if status == "CONFIRMED":
        confirmed = 1
    else:
        confirmed = 0

    curve = retreiveImageData.processCurve(kepID)
    retreiveImageData.plotCurve(curve, kepID, confirmed)
    currentIndex += 1
