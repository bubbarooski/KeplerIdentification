import lightkurve as lk
from matplotlib import pyplot as plt

folderName = "Lightcurves/"
fileExtension = ".png"


# Main function to process the curve
def processCurve(starID):
    curve = retrieveCurve(starID)
    curveDF = transformCurve(curve)
    return curveDF


# Retrieving and removing outliers via Lightkurve itself
def retrieveCurve(name):
    lcs = lk.search_lightcurve(name, author='Kepler', limit=10).download_all()
    lcRaw = lcs.stitch()
    lcClean = lcRaw.remove_outliers(sigma=20, sigma_upper=4)

    return lcClean


# Converting to dataframe and only keeping time and flux columns
def transformCurve(lightCurve):
    df = lk.LightCurve.to_pandas(lightCurve)
    df.reset_index(inplace=True)
    simpleLightCurve = df[['time', 'flux']]

    return simpleLightCurve


# Plotting
def plotCurve(lightCurve, kepID):
    plt.figure(figsize=(10,6))
    plt.axis('off')
    plt.plot(lightCurve.time, lightCurve.flux)

    # Saving figure
    export = folderName + kepID + fileExtension
    plt.savefig(export, format="png", dpi=500, bbox_inches='tight')
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
