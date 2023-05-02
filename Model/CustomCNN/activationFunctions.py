import numpy as np
from baseLayer import baseLayer
from activationLayer import activationLayer


class sigmoid(activationLayer):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoidPrime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoidPrime)


class Softmax(baseLayer):
    def forwardPropagation(self, inputValue):
        temp = np.exp(inputValue)
        self.outputValue = temp / np.sum(temp)
        return self.outputValue

    def backwardPropagation(self, outputGradient, learningRate):
        n = np.size(self.outputValue)
        return np.dot((np.identity(n) - self.outputValue.T) * self.outputValue, outputGradient)
