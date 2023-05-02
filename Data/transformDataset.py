import retreiveImageData
import pandas as pd


def dataDriver(name):
    file = readCSV(name)
    editCSV(file)


def readCSV(filename):
    celestialBodyList = pd.read_csv(filename)
    # print(celestialBodyList.head)

    return celestialBodyList


def editCSV(filename):
    editedCSV = filename[['kepid', 'koi_disposition']]
    # print(editedCSV.head())

    for index, row in editedCSV.iterrows():
        if row['koi_disposition'] == "CANDIDATE":
            editedCSV.drop(index, inplace=True)

    editedCSV.to_csv("keplerDataset.csv", index="false")

    return editedCSV
