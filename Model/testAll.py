import tensorflow as tf
from tensorflow.keras.models import load_model
import Model.tfCustomDataset as dataset
import numpy as np

CNN = load_model('tfCNN/keplerCNN.h5')
GRU = load_model('tfGRU/keplerGRU.h5')
MLP = load_model('tfMLP/keplerMLP.h5')

# print(test.get_weights())

print("Evaluating CNN:")
CNN.evaluate(dataset.datasetValidation, batch_size=10, verbose=2)

print("Evaluating GRU:")
GRU.evaluate(dataset.datasetValidation, batch_size=10, verbose=2)

print("Evaluating MLP:")
MLP.evaluate(dataset.datasetValidation, batch_size=10, verbose=2)
