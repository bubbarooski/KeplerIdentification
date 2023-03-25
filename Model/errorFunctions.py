import numpy as np


# Calculates binary cross entropy function using: -(1/n)sum(actual*log(predicted) + (1-actual)log(1-predicted))
def binaryCrossEntropy(yActual, yPredicted):
    return -np.mean(yActual * np.log(yPredicted) + (1-yActual) * np.log(1-yPredicted))


# Calculate binary cross entropy derivative using: (1/n)((1-actual)/(1-predicted) - (actual/predicted)
def binaryCrossEntropyPrime(yActual, yPredicted):
    return ((1 - yActual) / (1 - yPredicted)) - yActual / yPredicted / np.size(yActual)
