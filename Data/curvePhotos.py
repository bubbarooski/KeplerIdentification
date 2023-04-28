import retreiveImageData
import pandas as pd


def photoDriver(name):
    file = readCSV(name)
    editedFile = editCSV(file)


def readCSV(filename):
    celestialBodyList = pd.read_csv(filename)
    print(celestialBodyList.head)
    return celestialBodyList


def editCSV(filename):


    return editedCSV
