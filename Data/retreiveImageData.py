import lightkurve as lk
from matplotlib import pyplot as plt

confirmedFolder = r"C:/Users/shane/Documents/GitHub/KeplerIdentification/Data/Lightcurves/Confirmed/"
notConfirmedFolder = r"C:/Users/shane/Documents/GitHub/KeplerIdentification/Data/Lightcurves/Not Confirmed/"
fileExtension = ".png"


# Main function to process the curve
def processCurve(starID):
    curve = retrieveCurve(starID)
    curveDF = transformCurve(curve)
    return curveDF


# Retrieving and removing outliers via Lightkurve itself
def retrieveCurve(name):
    try:
        lcs = lk.search_lightcurve(name, author='Kepler', limit=3).download_all()
        lcRaw = lcs.stitch()
        lcClean = lcRaw.remove_outliers(sigma=20, sigma_upper=4)
        lcClean = lcClean.fill_gaps()
        lcClean = lcClean.flatten()
        lcClean = lcClean.bin()
    except:
        print("Bad curve")
        pass

    return lcClean


# Converting to dataframe and only keeping time and flux columns
def transformCurve(lightCurve):
    df = lk.LightCurve.to_pandas(lightCurve)
    df.reset_index(inplace=True)
    simpleLightCurve = df[['time', 'flux']]


    return simpleLightCurve


# Plotting
def plotCurve(lightCurve, kepID, type):
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
