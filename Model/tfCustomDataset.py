"""
tfCustomDataset
Purpose: used by all of the model to generate the datasets used for training and validation
"""

import tensorflow as tf
import random

# 1 equals not body
# 0 equals body

imageHeight = 438
imageWidth = 731
batch_size = 25

seed = random.randint(0, 100)

datasetTrain = tf.keras.preprocessing.image_dataset_from_directory(
    directory=r'C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\Lightcurves',
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(imageHeight, imageWidth),
    shuffle=True,
    validation_split=.2,
    subset="training",
    seed=seed
)

datasetValidation = tf.keras.preprocessing.image_dataset_from_directory(
    directory=r'C:\Users\shane\Documents\GitHub\KeplerIdentification\Data\Lightcurves',
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(imageHeight, imageWidth),
    shuffle=True,
    validation_split=.2,
    subset="validation",
    seed=seed
)

# Viewing info about test batch ----------------------
# trainingData = datasetTrain.as_numpy_iterator()
# batch = trainingData.next()
# print(batch[1])
# ----------------------------------------------------
