import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import sys
import argparse
import nltk
sys.path.append('..')

from models.categorization import Categorization
from models.predict_category import predict_article_category
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import pickle

stopwords = stopwords.words('english')
num_words = 1000 #top (most common) number of words for tokenizer

def load_bbc_data(tune_model):
    """Loads the BBC News dataset for categorization.
    
    Args:
        tune_model: True if to run the hyperparameter tuning, otherwise generate stored model. 
    """
    training_percentage = 0.75 # 75% used for training
    bbc_text = pd.read_csv("datasets/categorization/bbc-text.csv")
    training_size = int(len(bbc_text) * training_percentage)

    #Get top words
    get_top_words(bbc_text)

    categories = bbc_text['category'] #Label
    text = bbc_text['text'] #Article text

    text = remove_stop_words(text) #Remove stop words

    categories_train = categories[0: training_size]
    categories_test = categories[training_size:]

    article_train = text[0: training_size]
    article_test = text[training_size:]

    print("Length of article training data: ", len(article_train))
    print("Length of categories training data: ", len(categories_train))
    print("Length of article testing data: ", len(article_test))
    print("Length of categories testing data: ", len(categories_test))

    preprocess_bbc(article_train, article_test, categories_train, categories_test, tune_model)

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

def get_top_words(bbc_data):
    """Get top 20 words for each category, used to help with summarization task"""

    top_words = []
    
    bbc_data = bbc_data.apply(lambda x: [item for item in x if item not in stopwords])
    top_word = bbc_data.groupby('category')['text'].apply(lambda x: nltk.FreqDist(nltk.tokenize.word_tokenize(' '.join(x)))) #Group words by category
    print(top_word)

    top_words = top_word.head(20).agg({'a':lambda x: list(x)}) #Top 20 words
    for word in top_words:
        top_words.append(top_words)
    
    with open("top_category_words.txt", "wb") as save: #Save the top words, to be read in by summary model
        pickle.dump(top_words, save)

def preprocess_bbc(article_train, article_test, categories_train, categories_test, tune_model):
    """Preprocess bbc data for categorization
    - Tokenize, remove stop words, add padding, encode labels
    """
    tokenizer = Tokenizer(num_words=num_words) #Top x words (most common)
    tokenizer.fit_on_texts(article_train)
    word_dict = tokenizer.word_index 
    
    #Create training sequence
    train_sequences = tokenizer.texts_to_sequences(article_train) # x train

    #Create testing sequence
    test_sequences = tokenizer.texts_to_sequences(article_test) # x test

    #Padding ~ add 0s at end ('post') to make all 300
    train_padded = pad_sequences(sequences = train_sequences, maxlen=300, padding='post')
    test_padded = pad_sequences(sequences = test_sequences, maxlen=300, padding='post')

    #Encode labels (categories)
    encoder = LabelEncoder()
    encoder.fit(categories_train)
    categories_train_encoded = encoder.transform(categories_train)
    categories_test_encoded = encoder.transform(categories_test)

    encoder_mapping = dict(zip(encoder.classes_, categories_train_encoded)) # See the index assigned to each category
    print(encoder_mapping)

    #Display categories
    categories = encoder.classes_
    print(categories)

    total_categories = np.max(categories_train_encoded) + 1
    categories_train = keras.utils.to_categorical(categories_train_encoded, total_categories) # y train
    categories_test = keras.utils.to_categorical(categories_test_encoded, total_categories) # y test

    #Print shapes of train and test data for validation
    print(categories_train.shape)
    print(categories_test.shape)

    #Check article after tokenize and padding, and check it before (for validation)
    decoded_article = decode(train_padded[10], word_dict)
    print("Tokenized and padded article: ")
    print(decoded_article)
    print("---")
    print("Original article: ")
    print(article_train[10])

    #--- Pass data to Categorization model ---#
    model = Categorization(train_padded, test_padded, categories_test, categories_train)
    if (tune_model):
        model.run_tuning() # Find best HParams if '--tuning' parameter is passed.
    else:
        model.categorization_lstm(categories, encoder, categories_test_encoded)

def preprocess_article_for_categorization(article_text):
    """Handles the tokenization and padding of article text for categorization"""
    article = []
    article.append(article_text)

    #Tokenize
    tokenizer = Tokenizer(num_words=num_words) #Top 1000 words (most common)
    tokenizer.fit_on_texts(article)
    article_sequences = tokenizer.texts_to_sequences(article)

    #Padding
    article_padded = pad_sequences(sequences = article_sequences, maxlen=300)
    print(article_padded.shape) # Check shape
    return(predict_article_category(article_padded))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess BBC News data for categorization")
    parser.add_argument('--tuning', required=False, action='store_true', help='Run hyperparameter tuning to determine best')
    arguments = parser.parse_args()

    load_bbc_data(arguments.tuning)