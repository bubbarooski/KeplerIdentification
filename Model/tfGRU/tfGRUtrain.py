"""
tfGRUtrain
Purpose: Contains everything used to train the GRU and saves the model for future use
"""

import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

# Setting parameters
epochs = 50
batchSize = 10
inputShape = (dataset.imageHeight, dataset.imageWidth)

# Building model
keplerGRU = keras.models.Sequential()
keplerGRU.add(keras.Input(shape=inputShape))
keplerGRU.add(layers.GRU(128, activation='relu'))
keplerGRU.add(layers.Dense(32, activation='relu'))
keplerGRU.add(layers.Dense(10, activation='relu'))
keplerGRU.add(layers.Dense(1, activation='sigmoid'))
keplerGRU.build()
# print(keplerGRU.summary())

loss = keras.losses.BinaryCrossentropy()
optimization = keras.optimizers.SGD(learning_rate=.000001)
metrics = ["accuracy"]

keplerGRU.compile(optimizer=optimization, loss=loss, metrics=metrics)

hist = keplerGRU.fit(dataset.datasetTrain, epochs=epochs, batch_size=batchSize, verbose=2)

keplerGRU.evaluate(dataset.datasetValidation, batch_size=batchSize, verbose=2)

keplerGRU.save('keplerGRU.h5')
# print(keplerCNN.get_weights())

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
fig.suptitle('Accuracy vs Loss')
plt.legend(loc='upper right')
plt.show()
