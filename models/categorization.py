import tensorflow as tf
import numpy as np
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

if __name__ == "__main__":
    categorization_lstm()