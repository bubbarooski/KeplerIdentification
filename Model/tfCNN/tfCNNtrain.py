import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Model.tfCustomDataset as dataset

epochs = 50
batchSize = 5
inputShape = (dataset.imageHeight, dataset.imageWidth, 1)

keplerCNN = keras.models.Sequential()
keplerCNN.add(layers.Conv2D(32, 5, activation='relu', input_shape=inputShape))
keplerCNN.add(layers.MaxPool2D(3))
keplerCNN.add(layers.Conv2D(32, 5, activation='relu'))
keplerCNN.add(layers.MaxPool2D(3))
keplerCNN.add(layers.Flatten())
keplerCNN.add(layers.Dense(4, activation='relu'))
keplerCNN.add(layers.Dense(1, activation='sigmoid'))
keplerCNN.build()
# print(keplerCNN.summary())

loss = keras.losses.BinaryCrossentropy()
optimization = keras.optimizers.Adam(learning_rate=.0001)
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
