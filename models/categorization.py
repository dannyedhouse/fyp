import tensorflow as tf
from tensorflow import keras

# Hyperparameters for LSTM
input_length = 300 #length based on padding
epochs = 5 #Repetions
input_dim = 1000 # number of dimensions of features (top 1k words)
output_dim = 64

def categorization_lstm(train_padded, test_padded, categories_test, categories_train):
    """Create LTSM RNN for categorization"""

    model = tf.keras.Sequential()

    model.add(keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax')) # Num of categories

    print(model.summary()) # Show details of the model layers, shape and parameters

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(train_padded, categories_train, epochs=epochs, verbose=1, validation_split=0.1)

if __name__ == "__main__":
    categorization_lstm()