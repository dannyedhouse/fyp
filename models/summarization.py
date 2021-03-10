import tensorflow as tf
import numpy as np
import pandas as pd
import os
import re
import datetime
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from models.attention import AttentionLayer
from models.eval import evaluate
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

class Summarization:
    """Seq2Seq Model (encoder-decoder) for summarizing the news articles"""

    #Hyperparameters
    embedding_dim = 50
    output_dim = 50
    epochs = 5 #Repetions was 10
    batch_size = 64
    latent_dim = 256 #latent dimension
    dropout = 0.3
    
    def __init__(self, max_article_len, max_summary_len, reverse_target_word_index, reverse_source_word_index, target_word_index):
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len
        self.reverse_target_word_index = reverse_target_word_index
        self.reverse_source_word_index = reverse_source_word_index
        self.target_word_index = target_word_index
        

    def encoder(self):
        """Encoder"""
        encoder_input = keras.layers.Input(shape=(self.max_article_len,))
        
        # Embedding layer
        embedding = self.embedding_layer_articles(encoder_input)

        # First GRU (Gated Recurrent Unit)
        encoder_gru_1 = keras.layers.Bidirectional(keras.layers.GRU(self.latent_dim, return_sequences=True, return_state=True))
        encoder_output_1, encoder_forward_state_1, encoder_backward_state_1 = encoder_gru_1(embedding)
        encoder_output_dropout_1 = keras.layers.Dropout(self.dropout)(encoder_output_1)

        # Second GRU
        encoder_gru_2 = keras.layers.Bidirectional(keras.layers.GRU(self.latent_dim, return_sequences=True, return_state=True))
        encoder_output, encoder_forward_state, encoder_backward_state = encoder_gru_2(encoder_output_dropout_1)

        encoder_state = keras.layers.Concatenate()([encoder_forward_state, encoder_backward_state])

        return encoder_input, encoder_output, encoder_state


    def decoder(self, encoder_state):
        """Decoder"""
        decoder_input = keras.layers.Input(shape=(None, ))

        # Decoder embedding (summary embedding)
        decoder_embedding = self.embedding_layer_summaries(decoder_input)

        # initial state = encoder_states
        decoder_gru = keras.layers.GRU(self.latent_dim*2, return_sequences=True, return_state=True)

        decoder_output, decoder_state = decoder_gru(decoder_embedding, initial_state=[encoder_state])

        return decoder_input, decoder_output, decoder_state, decoder_gru

    def summarization_seq2seq(self, train_padded, test_padded, sum_train_padded, sum_test_padded, x_vocab, y_vocab, 
            embedding_matrix_x, embedding_matrix_y, test_articles):
        """Create Seq2Seq model for summarization using encoder, attention layer and decoder"""
        self.train_padded = train_padded
        self.test_padded = test_padded
        self.sum_train_padded = sum_train_padded 
        self.sum_test_padded = sum_test_padded
        self.x_vocab = x_vocab #Vocab length for the articles
        self.y_vocab = y_vocab #Vocab length for the summary
        self.embedding_matrix_x = embedding_matrix_x
        self.embedding_matrix_y = embedding_matrix_y
        self.test_articles = test_articles
        
        #Create embedding layers for article and summaries
        self.embedding_layer_articles = keras.layers.Embedding(self.x_vocab+1, self.embedding_dim, mask_zero=True,
            weights=[self.embedding_matrix_x], input_length=self.max_article_len, trainable=False)

        self.embedding_layer_summaries = keras.layers.Embedding(self.y_vocab+1, self.embedding_dim, mask_zero=True,
            weights=[self.embedding_matrix_y], input_length=self.max_summary_len, trainable=False)

        #-- Call Encoder --
        encoder_input, encoder_output, encoder_state = self.encoder()
        #Create encoder model for inference later
        encoder_model = keras.models.Model(encoder_input, [encoder_output, encoder_state])

        #-- Call Decoder --
        decoder_input, decoder_output, decoder_state, decoder_gru = self.decoder(encoder_state)

        #Attention layer
        attention_layer = AttentionLayer() 
        attention_output, attention_states = attention_layer([encoder_output, decoder_output])
        
        # Concat attention input and decoder LSTM output
        decoder_concat_input = keras.layers.Concatenate(axis=-1)([decoder_output, attention_output])

        # Dense layer
        decoder_dense = keras.layers.TimeDistributed(keras.layers.Dense(self.y_vocab+1, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        #Create decoder model for inference
        decoder_model = self.decoder_inference_model(decoder_gru, decoder_input, decoder_dense)

        model = keras.models.Model([encoder_input, decoder_input], decoder_outputs)
        print(model.summary())

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        es = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=1, min_delta=1e-4)

        history = model.fit([self.train_padded, self.sum_train_padded[:,:-1]], self.sum_train_padded[:,1:], epochs=self.epochs, batch_size=self.batch_size,
            validation_data=([self.test_padded, self.sum_test_padded[:,:-1]], self.sum_test_padded[:,1:]), workers=-1, callbacks=[es])

        #Accuracy and loss
        evaluate.display_accuracy_graph(history)
        evaluate.display_loss_graph(history)

        #Get summaries
        self.infer_summary(encoder_model, decoder_model)


        #---Evaluaton---#
        #Calculate BLEU/ROUGE
        self.calculate_metrics(encoder_model, decoder_model)
        

        #---Save Models---#
        #Encoder
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        save_model = encoder_model.to_json()
        with open(os.path.join(cwd, "encoder_model.json"), "w") as json_file:
            json_file.write(save_model)
        encoder_model.save_weights("encoder_weights.h5")

        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)
        save_model_decode = decoder_model.to_json()
        with open(os.path.join(cwd, "decoder_model.json"), "w") as json_file:
            json_file.write(save_model_decode)
        decoder_model.save_weights("decoder_weights.h5")

    def decoder_inference_model(self, decoder_gru, decoder_input, decoder_dense):
        """Create decoder model for predicting summaries"""
        # Previous time step
        decoder_state = keras.layers.Input(shape=(self.latent_dim*2, ))
        decoder_intermittent_state_input = keras.layers.Input(shape=(self.max_article_len, self.latent_dim*2))

        decoder_embedding_inference = self.embedding_layer_summaries(decoder_input)
        #predict next word in sequence. Initial state = previous time step
        decoder_output, decoder_state_inference = decoder_gru(decoder_embedding_inference, initial_state=[decoder_state])

        # Attention layer
        attention_layer = AttentionLayer()
        attention_output, attention_state = attention_layer([decoder_intermittent_state_input, decoder_output])
        decoder_inference_concat = keras.layers.Concatenate(axis=-1)([decoder_output, attention_output])

        #Dense layer
        decoder_output_inference = decoder_dense(decoder_inference_concat)

        # Create decoder model
        decoder_inference_output = keras.models.Model([decoder_input, decoder_intermittent_state_input, decoder_state], 
                                        [decoder_output_inference, decoder_state_inference])
        
        return decoder_inference_output

    def sequence_to_article(self, input_sequence):
        """Convert article sequence of indexes to the original article"""
        article_string = ''

        for item in input_sequence:
            if not item == 0:
                article_string = article_string + self.reverse_source_word_index[item] + ' '
        
        return article_string

    def sequence_to_summary(self, input_sequence):
        """Convert summary sequence of indexes to the original (target) summary"""
        summary_string = ''

        for item in input_sequence:
            if (not item == 0 and not item == self.target_word_index['start']) and not item == self.target_word_index['end']:
                summary_string = summary_string + self.reverse_target_word_index[item] + ' '

        return summary_string

    def search(self, sequence, token):
        """Search for token in a text sequence"""
        for i in range(len(sequence)):
            if sequence[i] == token:
                return True
        return False

    def infer_summary(self, encoder_model, decoder_model):
        """Get predicted and original summary of articles"""

        #Make predictions
        for i in range(0,8):
            print("Original article:", self.sequence_to_article(self.test_padded[i]))
            test_article = self.sequence_to_article(self.test_padded[i])
            test_article = re.sub('[^a-z]+', ' ', test_article)

            result = keras.preprocessing.text.text_to_word_sequence(test_article)

            if self.search(result, 'ukn'):
                index=result.index('ukn')
        
                input_org = re.sub('[^a-z]+',' ', self.test_articles[i])
                input_org = keras.preprocessing.text.text_to_word_sequence(input_org)
                ukn_token = input_org[index]

            else:
                ukn_token='ukn'

            print("ukn token:", ukn_token)
            print("Target summary:", self.sequence_to_summary(self.sum_test_padded[i]))
            print("Predicted summary:", self.decode_sequence(self.test_padded[i].reshape(1,self.max_article_len), encoder_model, decoder_model, ukn_token))
            print("\n")

    def decode_sequence(self, input_sequence, encoder_model, decoder_model, ukn_token):
        """Generate a summary for an article using models"""

        # Encode Input
        encoder_output, encoder_state = encoder_model.predict(input_sequence)

        # Target sequence of length = 1
        target_sequence = np.zeros((1, 1))

        # Choose 'start' as first word of target summary
        target_sequence[0, 0] = self.target_word_index['start']
        print("Target sequence: ", target_sequence)

        predicted_summary = ''
        summary_generated = False
        while not summary_generated:
            # Create predictions for next token
            token_output, decoder_state = decoder_model.predict([target_sequence, encoder_output, encoder_state])

            # Sample the next token from vocab (greedy search)
            token_index = np.argmax(token_output[0, -1, :])
            sampled_token = self.reverse_target_word_index[token_index]

            if not sampled_token == 'end':
                if (sampled_token=='ukn'):
                    predicted_summary += ' ' + ukn_token 
                
                else:
                    predicted_summary += ' ' + sampled_token

            # Break Condition: Reach max length of summaries or find 'end'
            if sampled_token == 'end' or len(predicted_summary.split()) >= (self.max_summary_len - 1):
                summary_generated = True

            # Update Target Sequence (length 1).
            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = token_index

            # Update internal states
            encoder_state = decoder_state

        return predicted_summary

    def calculate_metrics(self, encoder_model, decoder_model):
        """Calculates the BLEU and Rouge metrics for evaluating the summarization"""
        target_summaries = [] #reference
        summaries = [] #hypothesis
        bleu_scores = []
        
        for i in range(len(self.test_padded)):
            test_article = self.sequence_to_article(self.test_padded[i])
            test_article = re.sub('[^a-z]+', ' ', test_article)

            result = keras.preprocessing.text.text_to_word_sequence(test_article)

            if self.search(result, 'ukn'):
                index=result.index('ukn')
        
                input_org = re.sub('[^a-z]+',' ', self.test_articles[i])
                input_org = keras.preprocessing.text.text_to_word_sequence(input_org)
                ukn_token = input_org[index]

            else:
                ukn_token='ukn'

            target = self.sequence_to_summary(self.sum_test_padded[i])
            actual = self.decode_sequence(self.test_padded[i].reshape(1,self.max_article_len), encoder_model, decoder_model, ukn_token)

            target_summaries.append(target)
            summaries.append(actual)
            
            reference = [str(target).split()]
            candidate = str(actual).split()
            bleu_score = sentence_bleu(reference, candidate, weights=(1,0,0,0))
            bleu_scores.append(bleu_score)

        #ROUGE
        rouge = Rouge()
        scores = rouge.get_scores(target_summaries, summaries, avg=True)
        print(scores)

        #BLEU
        avg = sum(bleu_scores) / len(bleu_scores)
        print(avg)


    def generate_summary(self, article, article_padded):
        """Load in summary models to generate summary"""
        print(article_padded)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(cwd)

        #Read in encoder model
        encoder_model_file = open("encoder_model.json","r")
        read_encoder_model = encoder_model_file.read()
        encoder_model_file.close()

        encoder_model = tf.keras.models.model_from_json(read_encoder_model)
        encoder_model.load_weights("encoder_weights.h5")

        #Read in decoder model
        decoder_model_file = open("decoder_model.json","r")
        read_decoder_model = decoder_model_file.read()
        decoder_model_file.close()

        decoder_model = tf.keras.models.model_from_json(read_decoder_model, custom_objects={'AttentionLayer': AttentionLayer})
        decoder_model.load_weights("decoder_weights.h5")

        #Generate summary
        print("Original article:", self.sequence_to_article(article_padded[0]))
        test_article = self.sequence_to_article(article_padded[0])
        test_article = re.sub('[^a-z]+', ' ', test_article)
        result = keras.preprocessing.text.text_to_word_sequence(test_article)

        if self.search(result, 'ukn'):
            index=result.index('ukn')
    
            input_org = re.sub('[^a-z]+',' ', article)
            input_org = keras.preprocessing.text.text_to_word_sequence(input_org)
            ukn_token = input_org[index]

        else:
            ukn_token='ukn'

        print("Predicted summary:", self.decode_sequence(article_padded[0].reshape(1,self.max_article_len), encoder_model, decoder_model, ukn_token))
