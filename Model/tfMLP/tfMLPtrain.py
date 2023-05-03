"""
tfMLPtrain
Purpose: Contains everything used to train the MLP and saves the model for future use
"""

import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

# Setting parameters
epochs = 100
batchSize = 25
inputShape = (dataset.imageHeight, dataset.imageWidth)

# Building model
keplerMLP = keras.models.Sequential()
keplerMLP.add(layers.Flatten(input_shape=inputShape))
keplerMLP.add(layers.Dense(64, activation='relu'))
keplerMLP.add(layers.Dense(32, activation='relu'))
keplerMLP.add(layers.Dense(1, activation='sigmoid'))
keplerMLP.build()
# print(keplerCNN.summary())

loss = keras.losses.BinaryCrossentropy()
optimization = keras.optimizers.Adam(learning_rate=.0001)
metrics = ["accuracy"]

keplerMLP.compile(optimizer=optimization, loss=loss, metrics=metrics)

hist = keplerMLP.fit(dataset.datasetTrain, epochs=epochs, batch_size=batchSize, verbose=2)

keplerMLP.evaluate(dataset.datasetValidation, batch_size=batchSize, verbose=2)

keplerMLP.save('keplerMLP.h5')
# print(keplerCNN.get_weights())

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
fig.suptitle('Accuracy vs Loss')
plt.legend(loc='upper right')
plt.show()
