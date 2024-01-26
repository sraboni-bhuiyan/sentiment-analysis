# Analysis of customer satisfaction using sentiment analysis

### This study presents a comprehensive approach to gather and analyze sentiment from 1500 comments related to the "Lord of the Rings" series on YouTube.

- The ipynb file was created and run in the google colab. 

- To be able to run the code, all the necessary packages are called in the begining cell, these are as follows:
`NLTK`, `Textblob`, `sklearn`, `keras`, `googleapiclient`, `pickle`, `collections`, `pandas`, `numpy`

- The cell for downloading the comments from youtube, needs a google API key, so it was run one time, then saved as a csv file and loaded 
each time inside the program. 

- The data consists of **1500** rows, but in order for it to be balanced, **1302** of it was selected.

- There is a markdown before each cell that explains what the cell does.

- A pre-trained model (glove.6B.50d) is used. [link to download](https://nlp.stanford.edu/projects/glove/)

- four models (Naive Bayes, SVM, LSTM with TF-IDF vectorization, LSTM with word2vec vectorization) are employed. report of precision, 
`recall`, `f1 score` and `accuracy` is calculated for all the four models. 

- The models were saved in a pickle file, to be able to load them, you can use this code in python: (if loaded, no need for train cells to 
be run again.)
```
nb_classifier = load_model('naive_bayes.pickle')
best_estimator_svm = load_model('svm.pickle')
model_lstm = load_model('lstm_tfidf')
model_pretrained = load_model('lstm_pretrained.pickle')
tfidf_vectorizer = load_model('TF-IDF.pickle')
tokenizer = load_model('word2vec.pickle')
```

