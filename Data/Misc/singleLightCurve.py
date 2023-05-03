"""
singleLightCurve
Purpose: used to test feature extraction on light curves
"""

import pandas as pd
from Data.retreiveImageData import processCurve


def processSingleLightcurve(index):
    """
    processingSingleLightcurve: Drive function for feature extraction
        Parameters:
            index, index of keplerDataset
        Returns:
            curveDF, pandas dataframe of light curve

    """

    kic = "KIC "
    dataset = pd.read_csv(r"/Data/keplerDataset.csv")

    kepID = dataset['kepid'][index]
    data = []

    for index, row in dataset.iterrows():
        if dataset['kepid'][index] == kepID:
            disposition = dataset['koi_disposition'][index]


    kepID = str(kepID)

    data.append(kepID)
    data.append(disposition)

    kepID = kic + kepID
    print(kepID)
    print()
    curve = processCurve(kepID)

    maxFlux = curve['flux'].max()
    data.append(maxFlux)

    minFlux = curve['flux'].min()
    data.append(minFlux)

    meanFlux = curve['flux'].mean()
    data.append(meanFlux)

    p2pFlux = maxFlux - minFlux
    data.append(p2pFlux)

    varianceFlux = curve['flux'].var()
    data.append(varianceFlux)

    curveDF = pd.DataFrame([data], columns=['kepid', 'koi_disposition', 'maxFlux',
                                            'minFlux', 'meanFlux', 'p2pFlux', 'varianceFlux'])

    # print(curveDF.head())

    return curveDF
