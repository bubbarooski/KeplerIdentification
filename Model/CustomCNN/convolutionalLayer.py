import numpy as np
from baseLayer import baseLayer
from scipy import signal

'''
parameters:
    inputShape: tuple of 3 dimensions
    kernelSize: size of kernel
    depth: # of kernels
'''


class convolutionalLayer(baseLayer):
    def __init__(self, inputShape, kernelSize, depth):
        inputDepth, inputHeight, inputWidth = inputShape
        self.depth = depth
        self.inputShape = inputShape
        self.inputDepth = inputDepth
        self.outputShape = (depth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1)
        self.kernelShape = (depth, inputDepth, kernelSize, kernelSize)
        self.kernels = np.random.randn(*self.kernelShape)
        self.biases = np.random.randn(*self.outputShape)


    '''
    parameters: inputValue
    returns: outputValue
    Description: calculates the output value by using the following equation:
        Y = B + sum(X cross-correlated with K)
    '''
    def forwardPropagation(self, inputValue):
        self.inputValue = inputValue
        self.outputValue = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.inputDepth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return self.output


    '''
    parameters: outputGradient, learningRate
    returns: inputGradient
    Description: This method takes in dE/dY and calculates 3 things:
        1. dE/dK = X cross-correlated with dE/dY
        2. dE/dB = dE/dY
        3. dE/dX = sum(dE/dY convolved with K)
    '''
    def backwardPropagation(self, outputGradient, learningRate):
        kernelGradient = np.zeros(self.kernelShape)
        inputGradient = np.zeros(self.inputShape)

        for i in range(self.depth):
            for j in range(self.inputDepth):
                kernelGradient[i, j] = signal.correlate2d(self.inputValue[j], outputGradient[i], "valid")
                inputGradient[j] += signal.convolve2d(outputGradient[i], self.kernels[i, j], "full")

        self.kernels -= learningRate * kernelGradient
        self.biases -= learningRate * outputGradient
        return inputGradient
