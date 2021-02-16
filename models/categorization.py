import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorboard.plugins.hparams import api as hp

class Categorization:

    # Hyperparameters for LSTM
    input_dim = 1000 # number of dimensions of features (top 1k words)
    input_length = 300 #length based on padding
    epochs = 5 #Repetions   
    output_dim = 64
    batch_size = 16
    dropout = 0.2 #Dropout rate

    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.4))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    def __init__(self, train_padded, test_padded, categories_test, categories_train):
        self.train_padded = train_padded
        self.test_padded = test_padded
        self.categories_test = categories_test 
        self.categories_train = categories_train

    def categorization_lstm(self, categories, encoder, categories_test_encoded):
        """Create LSTM RNN for categorization"""

        print(self.train_padded[0])
        model = tf.keras.Sequential()

        model.add(keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(5, activation='softmax')) # Num of categories

        print(model.summary()) # Show details of the model layers, shape and parameters

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #Log model on TensorBoard
        log_dir = "../logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(self.train_padded, self.categories_train, epochs=self.epochs, verbose=1, validation_split=0.1, callbacks=[tensorboard_callback])
        test_model = model.evaluate(self.test_padded, self.categories_test, batch_size=self.batch_size, verbose=1)
        print('Accuracy: ', test_model[1])
        self.display_accuracy_graph(history)

        print(categories)
        #Make some predictions
        for i in range(10):
            predict = model.predict(np.array([self.test_padded[i]]))
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

    def display_accuracy_graph(self, history):
        """Plots a graph with the accuracy and val_accuracy to help determine best number of epochs"""
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

    def run_tuning(self):
        """Loop through and test different HParams for the categorization model"""
        """Ref: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams"""

        log_dir = "../logs/hparam_tuning/"

        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams=[self.HP_BATCH_SIZE, self.HP_DROPOUT, self.HP_OPTIMIZER],
                metrics=[hp.Metric('accuracy', display_name='Accuracy')],
            )

        run_num = 0
        for batch_size in self.HP_BATCH_SIZE.domain.values:
            for dropout in (self.HP_DROPOUT.domain.min_value, self.HP_DROPOUT.domain.max_value):
                for optimizer in self.HP_OPTIMIZER.domain.values:
                    hparams = {self.HP_BATCH_SIZE: batch_size, self.HP_DROPOUT: dropout, self.HP_OPTIMIZER: optimizer}
                    
                    run_name = "run-%d" % run_num
                    print('Run: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    self.run(log_dir + run_name, hparams)
                    run_num += 1

    def run(self, run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)
            test_model = self.hyperparameter_tuning(hparams)
            tf.summary.scalar('accuracy', test_model, step=1)

    def hyperparameter_tuning(self, hparams):
        """Used to evaluate the accuracy and loss of different hyperparameters"""

        model = tf.keras.Sequential()

        model.add(keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim, input_length=self.input_length))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(hparams[self.HP_DROPOUT]))
        model.add(keras.layers.Dense(5, activation='softmax')) # Num of categories

        model.compile(loss='categorical_crossentropy', optimizer=hparams[self.HP_OPTIMIZER], metrics=['accuracy'])

        #Log model on TensorBoard
        log_dir = "../logs/hparam_tuning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(self.train_padded, self.categories_train, epochs=self.epochs, verbose=1, validation_split=0.1, callbacks=[
            tensorboard_callback, hp.KerasCallback(log_dir, hparams)])
        
        test_model = model.evaluate(self.test_padded, self.categories_test, batch_size=hparams[self.HP_BATCH_SIZE], verbose=1)
        print('\tTest loss:', test_model[0])
        print('\tTest accuracy:', test_model[1])

        return test_model[1]