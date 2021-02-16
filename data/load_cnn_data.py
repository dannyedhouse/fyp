import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import csv
import sys
import re
import argparse
sys.path.append('..')

from models.categorization import Categorization
from models.predict_category import predict_article_category
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


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

def split_data(data):
    """Split by highlights (target summary) and the article"""

    articles = []
    summary = []

    for row in data.take(2):
        article = np.array(row['article']).tolist().decode('utf-8')
        highlight = np.array(row['highlights']).tolist().decode('utf-8')
        articles.append(clean_article(article))
        summary.append(clean_article(highlight))

    return articles, summary

def clean_article(article):
    article = article.replace('\n', ' ').replace('(CNN)', '').replace('--', '')
    article = re.sub('[^A-Za-z0-9.,]+', ' ', article)
    article = re.sub(r'\s+', ' ', article)

    return article.lower()

def preprocess_cnndm():
    """todo"""

if __name__ == "__main__":
    load_dm_cnn_data()