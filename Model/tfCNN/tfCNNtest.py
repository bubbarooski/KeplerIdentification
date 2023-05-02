import numpy as np
import tensorflow as t
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import Model.tfCustomDataset as dataset


def cnnAccuracy():
    print("CNN accuracy: ")
    test = load_model(r'C:\Users\shane\Documents\GitHub\KeplerIdentification\Model\tfCNN\keplerCNN.h5')
    test.evaluate(dataset.datasetValidation, batch_size=5, verbose=2)


def cnnTest(file, status):
    path = r"C:/Users/shane/Documents/GitHub/KeplerIdentification/Data/Lightcurves"
    model = load_model(r'C:\Users\shane\Documents\GitHub\KeplerIdentification\Model\tfCNN\keplerCNN.h5')

    if status == 1:
        filepath = path + "/Confirmed/" + file
    else:
        filepath = path + "/Not Confirmed/" + file

    curve = image.load_img(filepath, color_mode='grayscale', target_size=(438, 731))
    curveArray = image.img_to_array(curve)
    curveBatch = np.expand_dims(curveArray, axis=0)
    # curvePreprocessed = keras.preprocess_input(curveBatch)
    prediction = model.predict(curveBatch)

    print("CNN Prediction")
    if prediction < .5:
        print('Celestial body predicted with output ' + str(prediction))
        print()
    else:
        print('No celestial body predicted with output ' + str(prediction))
        print()
