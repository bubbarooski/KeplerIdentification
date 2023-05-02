import numpy as np
from baseLayer import baseLayer


class reshapeLayer(baseLayer):
    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape

    def forwardPropagation(self, inputValue):
        return np.reshape(inputValue, self.outputShape)

    def backwardPropagation(self, outputGradient, learningRate):
        return np.reshape(outputGradient, self.inputShape)