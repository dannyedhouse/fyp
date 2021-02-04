import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

# Hyperparameters for LSTM
input_length = 300 #length based on padding
epochs = 5 #Repetions
input_dim = 1000 # number of dimensions of features (top 1k words)
output_dim = 64
batch_size = 32

def categorization_lstm(train_padded, test_padded, categories_test, categories_train, categories, encoder, categories_test_encoded):
    """Create LSTM RNN for categorization"""

    model = tf.keras.Sequential()

    model.add(keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax')) # Num of categories

    print(model.summary()) # Show details of the model layers, shape and parameters

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(train_padded, categories_train, epochs=epochs, verbose=1, validation_split=0.1)

    test_model = model.evaluate(test_padded, categories_test, batch_size=batch_size, verbose=1)
    print('Accuracy: ', test_model[1])

    #Make some predictions
    for i in range(10):
        predict = model.predict(np.array([test_padded[i]]))
        prediction_label = categories[np.argmax(predict)]
        print("Predicted article category:" + prediction_label)
        print("Actual article category:" + encoder.inverse_transform(categories_test_encoded)[i])

    #Save model to make live predictions
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)
    save_model = model.to_json()
    with open(os.path.join(cwd, "categorization_model.json"), "w") as json_file:
        json_file.write(save_model)

    model.save_weights("weights.h5")

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

    categories = ['business','entertainment','politics','sport','tech']
    prediction_label = categories[np.argmax(predicted_category)]
    print("Predicted article as " + prediction_label)

if __name__ == "__main__":
    categorization_lstm()