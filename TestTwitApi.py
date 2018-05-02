import os, json
import pandas
import numpy
import sklearn
import sklearn.feature_extraction
import sklearn.model_selection 
import sklearn.metrics 
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network
from TwitterAPI import TwitterAPI, TwitterOAuth
import nltk
import nltk.corpus
import matplotlib.pyplot
from glob import glob

nltk.download('stopwords')
stopwds = list(nltk.corpus.stopwords.words('english'))

def print_score(Ytrue,Ypred):
  s = (sklearn.metrics.precision_score(Ytrue,Ypred),
          sklearn.metrics.recall_score(Ytrue,Ypred),
          sklearn.metrics.f1_score(Ytrue,Ypred))
  print('Precision: {:0.3}\nRecall: {:0.3}\nF-Score: {:0.3}\n'.format(*s))


cat_df = pandas.read_json("Data\\cat_tweet.json")
dog_df = pandas.read_json("Data\\dog_tweet.json")
catdog_df = pandas.read_json("Data\\catdog_tweet.json")

cat_df.head()
dog_df.head()
catdog_df.head()

cat_txt = [x.replace('#cat',"#blah") for x in cat_df['text']]
dog_txt = [x.replace('#dog',"#blah") for x in dog_df['text']]
catdog_txt = [(x.replace('#dogs',"#blah")).replace('#cat','#blah') for x in catdog_df['text']]
print(len(cat_txt))
print(len(dog_txt))
print(len(catdog_txt))
