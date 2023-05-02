import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

epochs = 100
batchSize = 10
inputShape = (dataset.imageHeight, dataset.imageWidth)

keplerGRU = keras.models.Sequential()
keplerGRU.add(keras.Input(shape=inputShape))
keplerGRU.add(layers.GRU(20, return_sequences=False, activation='relu'))
keplerGRU.add(layers.Dense(1, activation='sigmoid'))
keplerGRU.build()
# print(keplerGRU.summary())

loss = keras.losses.BinaryCrossentropy()
optimization = keras.optimizers.Adam(learning_rate=.0001)
metrics = ["accuracy"]

keplerGRU.compile(optimizer=optimization, loss=loss, metrics=metrics)

keplerGRU.fit(dataset.datasetTrain, epochs=epochs, batch_size=batchSize, verbose=2)

keplerGRU.evaluate(dataset.datasetValidation, batch_size=batchSize, verbose=2)

keplerGRU.save('keplerGRU.h5')
# print(keplerCNN.get_weights())
