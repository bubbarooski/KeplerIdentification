import retreiveImageData
import curvePhotos

print("Enter filename: ")

# KIC before input
# kepID = input()
# print(kepID)

# curve = retreiveImageData.processCurve(kepID)
# retreiveImageData.plotCurve(curve, kepID)


filename = input()
curvePhotos.photoDriver(filename)
