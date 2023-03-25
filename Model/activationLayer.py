import numpy as np
from baseLayer import baseLayer


class activationLayer(baseLayer):
    def __init__(self, activationFunction, activationFunctionPrime):
        self.activationFunction = activationFunction
        self.activationFunctionPrime = activationFunctionPrime


        '''
        parameters: inputValue
        returns: outputValue
        Description: This method is used to apply the activation function to the input value
        '''
        def forwardPropagation(self, inputValue):
            self.inputValue = inputValue
            return self.activationFunction(self.inputValue)


        '''
        parameters: outputGradient, learningRate
        returns: inputGradient
        Description: This method takes in dE/dY and 1) updates any trainable parameters and 2) updates the derivative
            of the error with respect to the input:
            dE/dX = (dE/dY) * f'(X) (element wise multiplication)
            The learning rate can be adjusted for parameter updating.
        '''
        def backwardPropagation(self, outputGradient, learningRate):
            return np.multiply((outputGradient, self.activationFunctionPrime(self.inputValue)))
