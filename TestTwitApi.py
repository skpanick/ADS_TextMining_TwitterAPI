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
tweet_count = 50
file_loc = "Data//"+str(tweet_count)+"//"

def print_score(Ytrue,Ypred):
  s = (sklearn.metrics.precision_score(Ytrue,Ypred),
          sklearn.metrics.recall_score(Ytrue,Ypred),
          sklearn.metrics.f1_score(Ytrue,Ypred))
  print('Precision: {:0.3}\nRecall: {:0.3}\nF-Score: {:0.3}\n'.format(*s))


cat_df = pandas.read_json(file_loc+"cat_tweet.json")
dog_df = pandas.read_json(file_loc+"dog_tweet.json")
catdog_df = pandas.read_json(file_loc+"catdog_tweet.json")

cat_df.head()
dog_df.head()
catdog_df.head()

cat_txt = [x.replace('#cat',"#blah") for x in cat_df['text']]
dog_txt = [x.replace('#dog',"#blah") for x in dog_df['text']]
catdog_txt = [(x.replace('#dogs',"#blah")).replace('#cat','#blah') for x in catdog_df['text']]
print(len(cat_txt))
print(len(dog_txt))
print(len(catdog_txt))

zeros = numpy.zeros((len(cat_txt),1))
ones = numpy.ones((len(dog_txt),1))
twos = numpy.ndarray(shape=(len(catdog_txt),1),dtype = int)
twos.fill(2)
print (zeros)
print ("\n")
print (ones)
print ("\n")
print (twos)
print ("\n")
Y = numpy.ravel(numpy.concatenate((zeros,ones,twos),axis=0))
print (Y)