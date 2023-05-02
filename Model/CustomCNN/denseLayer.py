from baseLayer import baseLayer
import numpy as np


class denseLayer(baseLayer):
    def __init__(self, inputSize, outputSize):
        # output x input sized matrix
        self.weights = np.random.rand(outputSize, inputSize)
        # single column matrix
        self.bias = np.random.randn(outputSize, 1)


    '''
    Forward propagation function
    parameters: inputValue
    returns: outputValue
    Description: This method takes in the input value and calculates the output value by applying the following
        function: y = wx+b, where w = weight, x = input, and b = bias
    '''
    def forwardPropagation(self, inputValue):
        self.inputValue = inputValue
        return np.dot(self.weight, self.input) + self.bias


    '''
    Backward propagation function
    parameters: outputGradient, learningRate
    returns: inputGradient
    Description: This method takes in dE/dY and calculates 3 things:
        1: dE/dW = (dE/dY) . inputValue.transposed
        2: dE/dB = (dE/dY)
        3: dE/dX = weights.transposed . dE/dY
        The learning rate is used for parameter tuning.
    '''
    def backwardPropogation(self, outputGradient, learningRate):
        weightGradient = np.dot(outputGradient, self.inputValue.T)
        self.weights -= learningRate * weightGradient
        self.bias -= learningRate * outputGradient
        return np.dot(self.weights.T, outputGradient)
