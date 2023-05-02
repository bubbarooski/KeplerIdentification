import retreiveImageData
import transformDataset

print("Enter filename: ")

# KIC before input
kepID = input()
print(kepID)

curve = retreiveImageData.processCurve(kepID)
retreiveImageData.plotCurve(curve, kepID)


# curvePhotos.photoDriver("keplerTrainingData.csv")
