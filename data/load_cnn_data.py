import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import spacy
import csv
import sys
import os
import re
import pickle
import argparse
sys.path.append('..')

from models.summarization import Summarization
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

stopwords = stopwords.words('english')
max_article_len = 650
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
        #Remove Stopwords
        article = clean_article(article)
        highlight = clean_article(highlight)
        articles.append(remove_stop_words(article))
        summary.append(remove_stop_words(highlight))

    print("The average length is: ",np.average(articles_length))
    print("The average length is: ",np.average(summary_length))

    return articles, summary

def clean_article(article):
    article = article.replace('\n', ' ').replace('(CNN)', '').replace('--', '')
    article=re.sub(r'>',' ', article)
    article=re.sub(r'<',' ', article)
    article=re.sub(r'LRB',' ', article)
    article=re.sub(r'RRB',' ', article)
    article = re.sub(r'[" "]+', " ", article)
    article=re.sub(r"([?!¿])", r" \1 ", article)
    article=re.sub(r'-',' ', article)
    article=article.replace('/',' ')
    article=re.sub(r'\s+', ' ', article)
    article=decontract(article)
    article = re.sub('[^A-Za-z0-9.,]+', ' ', article)
    article = re.sub(r'\s+', ' ', article)

    return article.lower()

def decontract(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'m", " am", phrase)  
    phase = re.sub(r"ain\'t", "is not", phrase) 
    return phrase
    
def remove_stop_words(article):
    """Remove stop words (using nltk corpus)"""
    word_token = word_tokenize(article)
    return ' '.join([word for word in word_token if word not in stopwords])
    
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

def preprocess_cnn_dm(train_articles_all, train_summary_all, test_articles, test_summary):
    """Preprocess cnn/dm data for summarization
    - Tokenization, padding
    """
    #------------------
    # Article tokenization + padding
    #------------------
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_articles_all)

    # Define rare words (below threshold) as 'ukn'
    threshold=2
    rare_words=[]
    for key,value in tokenizer.word_counts.items():
        if(value<threshold):
            rare_words.append(key)
    
    print("Num of rare words for articles:", len(rare_words))
    rare_words[:5]
    
    token_rare=[]
    for i in range(len(rare_words)):
        token_rare.append('ukn')
    
    rare_words_dict = dict(zip(rare_words,token_rare))

    train_articles=[]
    for i in train_articles_all:
        for word in i.split():
            if word.lower() in rare_words_dict:
                i = i.replace(word, rare_words_dict[word.lower()])
        train_articles.append(i)

    tokenizer = Tokenizer(oov_token='ukn')
    tokenizer.fit_on_texts(list(train_articles))

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
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(train_summary_all)
    
    # Define rare words as 'ukn'
    threshold = 2
    rare_words=[]
    for key, value in y_tokenizer.word_counts.items():
        if(value<threshold):
            rare_words.append(key)
    
    rare_words[3:10]

    token_rare=[]
    for i in range(len(rare_words)):
        token_rare.append('ukn')

    rare_words_dict = dict(zip(rare_words,token_rare))

    train_summary=[]
    for i in train_summary_all:
        for word in i.split():
            if word.lower() in rare_words_dict:
                i = i.replace(word, rare_words_dict[word.lower()])
        train_summary.append(i)

    print(train_summary_all[5])
    print(train_summary[5])
    y_tokenizer = Tokenizer(oov_token='ukn') 
    y_tokenizer.fit_on_texts(list(train_summary))

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
    # GloVe Pre-trained Word Embeddings
    #------------------  
    embeddings_dictionary = dict()
    glove_embeddings = open("glove/glove.6B.50d.txt",encoding="utf8")

    for line in glove_embeddings:
        line = line.split()
        word = line[0]
        vector_dimensions = np.asarray(line[1:],dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_embeddings.close()

    # Embeddings for article vocab
    embedding_matrix_x = np.zeros((article_vocab_size+1 , 50))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix_x[index] = embedding_vector

    # Embeddings for summary vocab
    embedding_matrix_y = np.zeros((summary_vocab_size+1, 50))
    for word, index in y_tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix_y[index] = embedding_vector

    #------------------
    # Call summarization model
    #------------------
    reverse_target_word_index=y_tokenizer.index_word
    reverse_source_word_index=tokenizer.index_word
    target_word_index=y_tokenizer.word_index
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('y_tokenizer.pickle', 'wb') as handle:
        pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = Summarization(max_article_len, max_summary_len, reverse_target_word_index, reverse_source_word_index, target_word_index)
    model.summarization_seq2seq(train_padded, test_padded, sum_train_padded, sum_test_padded, article_vocab_size, summary_vocab_size, 
        embedding_matrix_x, embedding_matrix_y, test_articles)


def preprocess_article_summary(article_text):
    """Preprocesses article for summarization"""
    article = []   
    article_text = clean_article(article_text)
    article.append(remove_stop_words(article_text))

    #Fit article to tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(article)
    article_sequences = tokenizer.texts_to_sequences(article)

    article_padded = pad_sequences(sequences = article_sequences, maxlen=max_article_len, padding='post')

    #Load tokenizers
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)
    
    with open('tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)

    with open('y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    model = Summarization(max_article_len, max_summary_len, reverse_target_word_index, reverse_source_word_index, target_word_index)
    print(model.generate_summary(article, article_padded))

if __name__ == "__main__":
    load_dm_cnn_data()