"""
retrieveImageData
Purpose: Contains all of the functions used to process a Kepler ID and retrieve its light curve.
"""

import lightkurve as lk
from matplotlib import pyplot as plt

confirmedFolder = r"C:/Users/shane/Documents/GitHub/KeplerIdentification/Data/Lightcurves/Confirmed/"
notConfirmedFolder = r"C:/Users/shane/Documents/GitHub/KeplerIdentification/Data/Lightcurves/Not Confirmed/"
fileExtension = ".png"


def processCurve(starID):
    """
    processCurve: Driver function for processing and retrieving curve for training data
        Parameters:
            starID, valid Kepler ID in form "KIC ######"
        Returns:
            curveDF, pandas dataframe of light curve
    """

    curve = retrieveCurve(starID)
    curveDF = transformCurve(curve)
    return curveDF


def retrieveCurve(starID):
    """
    retrieveCurve: searches via lightkurve API for curve and processes curve
        Parameters:
            starID, valid Kepler ID in form "KIC ######"
        Returns:
            lcClean, a clean light curve object
    """

    try:
        lcs = lk.search_lightcurve(starID, author='Kepler', limit=3).download_all()
        lcRaw = lcs.stitch()
        lcClean = lcRaw.remove_outliers(sigma=20, sigma_upper=4)
        lcClean = lcClean.fill_gaps()
        lcClean = lcClean.flatten()
        lcClean = lcClean.bin()
    except:
        print("Bad curve")
        pass

    return lcClean


def transformCurve(lightCurve):
    """
    transformCurve: transforms a light curve object into a dataframe consisting of time and
    flux only
        Parameters:
            lightCurve, a light curve object
        Returns:
            simpleLightCurve, pandas dataframe of light curve
    """

    df = lk.LightCurve.to_pandas(lightCurve)
    df.reset_index(inplace=True)
    simpleLightCurve = df[['time', 'flux']]


    return simpleLightCurve


def plotCurve(lightCurve, kepID, type):
    """
    plotCurve: creates plot of curve and saves it to corresponding directory
        Parameters:
            lightCurve, a light curve object
            kepID, valid Kepler ID in form "KIC ######"
            type, flag indicating confirmed or not confirmed
        Returns:
            void
    """

    plt.figure(figsize=(1,.5))
    plt.axis('off')
    plt.plot(lightCurve.time, lightCurve.flux, lw=.25, color='k')

    # Saving figure
    if type == 1:
        export = confirmedFolder + kepID + fileExtension
        plt.savefig(export, format="png", dpi=750, bbox_inches='tight')
        plt.close()
    else:
        export = notConfirmedFolder + kepID + fileExtension
        plt.savefig(export, format="png", dpi=750, bbox_inches='tight')
        plt.close()



def plotCurvePretty(lightCurve):
    """
    plotCurvePretty: creates nicer plot of light curve and displays it to user
        Parameters:
            lightCurve, a light curve object
        Returns:
            void
    """

    plt.figure(figsize=(6, 4))
    # plt.axis('off')
    plt.title("Light Curve")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.plot(lightCurve.time, lightCurve.flux, lw=1, color='k')
    plt.show()



# Tried messing with changing the scaling to account for different types of curves ------
# maxTime = simpleLightCurve['time'].max()
# minTime = simpleLightCurve['time'].min()
# xp = [minTime, maxTime]
# fp = [0, 10000]
# for ind in simpleLightCurve.index:
#     simpleLightCurve["time"][ind] = np.interp(simpleLightCurve["time"][ind], xp, fp)
# ---------------------------------------------------------------------------------------
# print(simpleLightCurve.info())
# simpleLightCurve.to_csv('test.csv')
