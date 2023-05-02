from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

epochs = 100
batchSize = 25
inputShape = (dataset.imageHeight, dataset.imageWidth)

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

keplerMLP.fit(dataset.datasetTrain, epochs=epochs, batch_size=batchSize, verbose=2)

keplerMLP.evaluate(dataset.datasetValidation, batch_size=batchSize, verbose=2)

keplerMLP.save('keplerMLP.h5')
# print(keplerCNN.get_weights())
