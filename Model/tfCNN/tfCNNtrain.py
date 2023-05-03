"""
tfCNNtrain
Purpose: Contains everything used to train the CNN and saves the model for future use
"""

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

# Setting parameters
epochs = 50
batchSize = 5
inputShape = (dataset.imageHeight, dataset.imageWidth)

# Building model
keplerCNN = keras.models.Sequential()
keplerCNN.add(layers.Conv1D(128, 30, activation='relu', input_shape=inputShape))
keplerCNN.add(layers.MaxPool1D(10))
keplerCNN.add(layers.Conv1D(32, 30, activation='relu'))
keplerCNN.add(layers.MaxPool1D(10))
keplerCNN.add(layers.Flatten())
keplerCNN.add(layers.Dense(10, activation='relu'))
keplerCNN.add(layers.Dense(1, activation='sigmoid'))
keplerCNN.build()
# print(keplerCNN.summary())

loss = keras.losses.BinaryCrossentropy()
optimization = keras.optimizers.Adam(learning_rate=.00001)
metrics = ["accuracy"]

keplerCNN.compile(optimizer=optimization, loss=loss, metrics=metrics)

hist = keplerCNN.fit(dataset.datasetTrain, epochs=epochs, batch_size=batchSize, verbose=2)

keplerCNN.evaluate(dataset.datasetValidation, batch_size=batchSize, verbose=2)

keplerCNN.save('keplerCNN.h5')
# print(keplerCNN.get_weights())

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
fig.suptitle('Accuracy vs Loss')
plt.legend(loc='upper right')
plt.show()
