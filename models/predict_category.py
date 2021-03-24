import tensorflow as tf
import numpy as np
import os

def predict_article_category(article):
    """Loads the categorization model using the generated json file to make live prediction"""

    #Read in model
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)
    model_file = open("categorization_model.json","r")
    read_model = model_file.read()
    model_file.close()

    categorization_model = tf.keras.models.model_from_json(read_model)
    categorization_model.load_weights("weights.h5")
    categorization_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Make prediction
    predicted_category = categorization_model.predict(article)

    categories = ['business','entertainment','tech','sport','politics']
    prediction_label = categories[np.argmax(predicted_category)]
    print("Predicted article as " + prediction_label)
    return prediction_label.capitalize()