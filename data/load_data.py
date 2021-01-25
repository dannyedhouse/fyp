import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import csv
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
stopwords = stopwords.words('english')

def load_dm_cnn_data():
    """TO DO"""
    data = tfds.load('cnn_dailymail', data_dir="~/tensorflow_datasets")
    train_data = data['train']
    test_data = data['test']

def load_bbc_data():
    """Loads the BBC News dataset for categorization"""

    training_percentage = 0.7 # Try 70/30 split

    bbc_text = pd.read_csv("datasets/categorization/bbc-text.csv")
    training_size = int(len(bbc_text) * training_percentage)

    categories = bbc_text['category'] #Label
    text = bbc_text['text'] #Article text

    print(text[10])
    text = remove_stop_words(text)
    print('\n', text[10])

    categories_train = categories[0: training_size]
    categories_test = categories[training_size:]

    article_train = text[0: training_size]
    article_test = text[training_size:]

    print("Length of article training data: ", len(article_train))
    print("Length of categories training data: ", len(categories_train))
    print("Length of article testing data: ", len(article_test))
    print("Length of categories testing data: ", len(categories_test))

    preprocess_bbc(article_train, article_test, categories_train, categories_test)

def remove_stop_words(article):
    """Remove stop words (using nltk corpus)"""
    return article.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

def decode(article, word_dict):
    """Decode an article word index to get article content"""

    get_word_dict = dict([ (value, key) for (key, value) in word_dict.items() ])
    decoded = " "

    for i in article:
        decoded += " " + get_word_dict.get(i, 'x') # put 'x' if word cant be found in index, i.e. not in top 1000 words
    
    return decoded

def preprocess_bbc(article_train, article_test, categories_train, categories_test):
    """Preprocess bbc data for categorization
    - Tokenize, remove stop words, add padding, encode labels
    """
    tokenizer = Tokenizer(num_words=1000) #Top 1000 words
    tokenizer.fit_on_texts(article_train)
    word_dict = tokenizer.word_index 
    print(len(word_dict))
    
    #Create training sequence
    train_sequences = tokenizer.texts_to_sequences(article_train) # x train
    print(train_sequences[10])

    #Create testing sequence
    test_sequences = tokenizer.texts_to_sequences(article_test) # x test

    #Padding ~ add 0s at end ('post') to make all 300
    train_padded = pad_sequences(sequences = train_sequences, maxlen=300, padding='post')
    print(len(train_sequences[0]))
    print(len(train_padded[0]))
    print(train_padded[0])

    test_padded = pad_sequences(sequences = test_sequences, maxlen=300, padding='post')
    print(test_padded.shape)

    #Encode labels (categories)
    encoder = LabelEncoder()
    encoder.fit(categories_train)
    categories_train_encoded = encoder.transform(categories_train)
    categories_test_encoded = encoder.transform(categories_test)

    total_categories = np.max(categories_train_encoded) + 1
    categories_train = keras.utils.to_categorical(categories_train_encoded, total_categories) # y train
    categories_test = keras.utils.to_categorical(categories_test_encoded, total_categories) # y test

    print(categories_train[0])
    print(categories_train.shape)
    print(categories_test.shape)

    #Check article after tokenize and padding, and check it before (for validation)
    decoded_article = decode(train_padded[10], word_dict)

    print("Tokenized and padded article: ")
    print(decoded_article)
    print("Original article: ")
    print(article_train[10])

if __name__ == "__main__":
    load_bbc_data()