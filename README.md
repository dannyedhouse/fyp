# News Bites - Final Year Project [WIP]

Combining Categorization and Summarization of online news articles using NLP and Machine Learning.

## Usage

`py load_data.py` - Load and preprocess BBC News dataset and generate Categorization model

`py load_data.py --tuning` - Run experiments with different hyperparameters (HParam) to determine best for Categorization model.  Then run `tensorboard --logdir="logs/"` to  view the accuracy results of the different parameters.