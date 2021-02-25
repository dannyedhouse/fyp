import tensorflow as tf
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp

class Summarization:
    """Seq2Seq Model (encoder-decoder) for summarizing the news articles"""

    #Hyperparameters
    output_dim = 50
    epochs = 5 #Repetions   
    batch_size = 64
    latent_dim = 128
    
    def __init__(self, train_padded, test_padded, sum_train_padded, sum_test_padded, max_article_len, max_summary_len, x_vocab, y_vocab):
        self.train_padded = train_padded
        self.test_padded = test_padded
        self.sum_train_padded = sum_train_padded 
        self.sum_test_padded = sum_test_padded
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len
        self.x_vocab = x_vocab #Vocab length for the articles
        self.y_vocab = y_vocab #Vocab length for the summary

    def encoder(self):
        """Encoder"""
        encoder_inputs = keras.layers.Input(shape=(self.max_article_len,))

        #Embedding Layer
        embedding = keras.layers.Embedding(input_dim = self.x_vocab+1, output_dim = self.output_dim, mask_zero=True, trainable=True)(encoder_inputs)
    
        #Bidirectional Layer
        encoder_lstm = keras.layers.Bidirectional(keras.layers.LSTM(64, return_state=True, dropout=0.4))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(embedding)

        #Concatenate forward and backward cells (used by decoder for initial states)
        state_h = keras.layers.Concatenate()([forward_h, backward_h])
        state_c = keras.layers.Concatenate()([forward_c, backward_c])
        
        return encoder_inputs, encoder_outputs, state_h, state_c

    def decoder(self, state_h, state_c, encoder_inputs):
        """Decoder"""
        decoder_inputs = keras.layers.Input(shape=(self.max_summary_len-1,))

        #Embedding Layer
        decoder_embedding = keras.layers.Embedding(input_dim = self.y_vocab+1, output_dim = self.output_dim, mask_zero=True, trainable=True)
        decoder_embedded = decoder_embedding(decoder_inputs)

        #LSTM
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True, dropout=0.4)
        decoder_output, decoder_state_h, decoder_state_c = decoder_lstm(decoder_embedded, initial_state=[state_h,state_c])

        return decoder_inputs, decoder_output

    def summarization_seq2seq(self):
        """Create Seq2Seq model for summarization using encoder, attention layer and decoder"""

        print(self.x_vocab)
        print(self.y_vocab)

        #Call encoder
        encoder_inputs, encoder_outputs, state_h, state_c = self.encoder()
        #Call decoder
        decoder_inputs, decoder_output = self.decoder(state_h, state_c, encoder_inputs)

        # Dense layer
        decoder_dense =  keras.layers.TimeDistributed(keras.layers.Dense(self.y_vocab+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_output)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5)

        history = model.fit([self.train_padded, self.sum_train_padded[:,:-1]], self.sum_train_padded[:,1:], epochs=self.epochs, batch_size=self.batch_size,
            validation_data=([self.test_padded, self.sum_test_padded[:,:-1]], self.sum_test_padded[:,1:]), workers=-1, callbacks=[es])

