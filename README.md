# News Bites - Final Year Project

Combining Categorization and Summarization of online news articles using NLP and Machine Learning.

# Usage Guide

## Set up environment

1. Create a virtualenv using `python -m venv env`
2. Activate the environment `.\env\Scripts\activate`
3. Install pip requirements with `pip install -r .\requirements.txt`

## Categorization

The categorization model uses the BBC News classification dataset (csv) of 2,225 articles with 5 categories; Business, Entertainment, Politics, Sport and Tech. 

This must be downloaded (e.g. from [here](https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv)) and placed into the *data/datasets/categorization* folder

`py train_bbc_data.py` - Load and preprocess BBC News dataset, train the Categorization model and save the model as a .json file, with the weights as a HDF file to be loaded in later.

`py load_bbc_data.py --tuning` - Run experiments with different hyperparameters (HParam) to determine best for Categorization model.  Then run `tensorboard --logdir="logs/"` to  view the accuracy results of the different parameters.

## Summarization

The summarization model uses the CNN/DailyMail dataset, which is loaded in using Tensorflow Datasets. 

The dataset consists of 287,000 documents with the article and highlight (target summary). As such, it can take a while to train the model, therefore the *summarization_model.ipynb* Jupyter Notebook file may be run using Google colab with a GPU to increase training performance.
- Note this requries the glove.6B.50d.txt GloVe embeddings to located at */content/gdrive/MyDrive/glove/glove.6B.50d.txt*

Otherwise, `py train_cnn_data.py` will load and preprocess the dataset, and train the model.

#### Word Embeddings
The summarization model uses pre-trained GloVe word embeddings, glove.6B.50d.txt (which can be downloaded from [here](https://www.kaggle.com/watts2/glove6b50dtxt)), and must be located in *data/glove*

## API
Run `api.py` in order to start the Flask API, which will be hosted at localhost.
The endpoint **/summary** takes the parameter **url** followed by the article link to be summarized.
- e.g. */summary?url=https://www.bbc.co.uk/news/uk-55489932*

This returns the *category*, *imageURL*, *summary* and *title* of the article.

## Unit Tests
Run `python -m unittest discover test` to execute the unit tests located in the /test folder.