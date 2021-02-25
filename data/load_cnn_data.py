import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import spacy
import csv
import sys
import re
import argparse
sys.path.append('..')

from models.summarization import Summarization
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

max_article_len = 800
max_summary_len = 55

def load_dm_cnn_data():
    """Load CNN/DM dataset for Summarization."""
    data = tfds.load('cnn_dailymail', data_dir="~/tensorflow_datasets")

    train_data = data['train']
    test_data = data['test']

    print(data) #PrefetchDataset

    train_articles, train_summary = split_data(train_data)
    test_articles, test_summary = split_data(test_data)

    print("Length of article training data: ", len(train_articles))
    print("Length of summary training data: ", len(train_summary))
    print("Length of article testing data: ", len(test_articles))
    print("Length of summary training data: ", len(test_summary))

    print(train_articles[1])
    print(train_summary[1])

    preprocess_cnn_dm(train_articles, train_summary, test_articles, test_summary)

def split_data(data):
    """Split by highlights (target summary) and the article"""

    articles = []
    summary = []

    articles_length = []
    summary_length = []
    
    for row in data.take(100):
        article = np.array(row['article']).tolist().decode('utf-8')
        highlight = np.array(row['highlights']).tolist().decode('utf-8')
        articles_length.append(count_words(article))
        summary_length.append(count_words(highlight))
        #named_entity_recognition(article)

        articles.append(clean_article(article))
        summary.append(clean_article(highlight))

    print("The average length is: ",np.average(articles_length))
    print("The average length is: ",np.average(summary_length))

    return articles, summary

def clean_article(article):
    article = article.replace('\n', ' ').replace('(CNN)', '').replace('--', '')
    article = re.sub('[^A-Za-z0-9.,]+', ' ', article)
    article = re.sub(r'\s+', ' ', article)

    return article.lower()

def count_words(article):
    """Return number of words in an article/summary"""
    split = article.split()
    num_words = len(split)
    return num_words

def named_entity_recognition(article):
    """Subtitute named entities (names, companies, locations etc...) with the entity category"""

    #print(article)
    #print("---")
    ner = nlp(article)
    new = (" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in ner]))
    new=new.lower()
    #print(new)

def preprocess_cnn_dm(train_articles, train_summary, test_articles, test_summary):
    """Preprocess cnn/dm data for summarization
    - Tokenization, padding
    """
    #------------------
    # Article tokenization + padding
    #------------------
    tokenizer = Tokenizer(oov_token='ukn')
    tokenizer.fit_on_texts(train_articles)

    #Create training sequence
    train_sequences = tokenizer.texts_to_sequences(train_articles) # x train
    print(train_sequences[0])

    #Create testing sequence
    test_sequences = tokenizer.texts_to_sequences(test_articles) # x test
    
    #Padding
    train_padded = pad_sequences(sequences = train_sequences, maxlen=max_article_len, padding='post')
    print(len(train_sequences[0]))
    print(len(train_padded[0]))
    print(train_padded[0])

    test_padded = pad_sequences(sequences = test_sequences, maxlen=max_article_len, padding='post')
    print(test_padded.shape)

    article_vocab_size = len(tokenizer.word_index) + 1

    #------------------
    # Summary tokenization + padding
    #------------------
    y_tokenizer = Tokenizer(oov_token='ukn')
    y_tokenizer.fit_on_texts(train_summary)

    #Create training sequence
    sum_train_sequences = y_tokenizer.texts_to_sequences(train_summary) # x train
    print(sum_train_sequences[0])

    #Create testing sequence
    sum_test_sequences = y_tokenizer.texts_to_sequences(test_summary) # x test
    
    #Padding
    sum_train_padded = pad_sequences(sequences = sum_train_sequences, maxlen=max_summary_len, padding='post')
    sum_test_padded = pad_sequences(sequences = sum_test_sequences, maxlen=max_summary_len, padding='post')
    print(sum_test_padded.shape)

    summary_vocab_size = len(y_tokenizer.word_index) + 1

    #------------------
    # Call summarization model
    #------------------
    model = Summarization(train_padded, test_padded, sum_train_padded, sum_test_padded, max_article_len, max_summary_len, 
        article_vocab_size, summary_vocab_size)
    model.summarization_seq2seq()

if __name__ == "__main__":
    load_dm_cnn_data()