class baseLayer:
    def __init__(self):
        self.inputValue = None
        self.outputValue = None


    '''
    Forward propagation function
        parameters: inputValue
        returns: outputValue
        Description: This method is used to take in an input and calculate an output
    '''
    def forwardPropagation(self, inputValue):
        pass


    '''
    Backward propagation function
        parameters: outputGradient, learningRate
        returns: inputGradient
        Description: This method takes in dE/dY and 1) updates any trainable parameters and 2) updates the derivative
            of the error with respect to the input. The learning rate can be adjusted for parameter updating.
    '''
    def backwardPropagation(self, outputGradient, learningRate):
        pass
