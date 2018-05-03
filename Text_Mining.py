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
import nltk
import nltk.corpus
import matplotlib.pyplot
from glob import glob

nltk.download('stopwords')
stopwds = list(nltk.corpus.stopwords.words('english'))
tweet_count = 4000
file_loc = "Data//"+str(tweet_count)+"//"
prefix_list = ["cat","dog","catdog"]

def print_score(Ytrue,Ypred):
	prescision = sklearn.metrics.precision_score(Ytrue,Ypred,average=None)
	recall = sklearn.metrics.recall_score(Ytrue,Ypred,average=None)
	f1_score = sklearn.metrics.f1_score(Ytrue,Ypred,average=None)
	prescision.tolist()
	recall.tolist()
	f1_score.tolist()
	for i in range(len(prefix_list)):
		print("Scores for " , prefix_list[i])
		print ("Prescision :",prescision[i])
		print ("Recall :",recall[i])
		print ("F1_Score :",f1_score[i])
		print("\n")

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
vectorizer = sklearn.feature_extraction.text.CountVectorizer(cat_txt+dog_txt+catdog_txt, analyzer='word', stop_words=stopwds, min_df=5)
vectorizer.fit(cat_txt+dog_txt+catdog_txt)
cat_tdm = vectorizer.transform(cat_txt).toarray()
dog_tdm = vectorizer.transform(dog_txt).toarray()
catdog_tdm = vectorizer.transform(catdog_txt).toarray()

"""### Create visible matricies and combine"""
zeros = numpy.zeros((len(cat_txt),1))
ones = numpy.ones((len(dog_txt),1))
twos = numpy.ndarray(shape=(len(catdog_txt),1),dtype = int)
twos.fill(2)
combined_tdm = numpy.concatenate((cat_tdm,dog_tdm,catdog_tdm),axis=0)
Y = numpy.ravel(numpy.concatenate((zeros,ones,twos),axis=0))

"""### Create train/test split for modeling"""
Xtrain,Xtest,Ytrain,Ytest = sklearn.model_selection.train_test_split(combined_tdm, Y, test_size=.20)

#print (Xtrain)

"""### Naive Bayes"""
nb = sklearn.naive_bayes.GaussianNB()
#nb = sklearn.naive_bayes.MultinomialNB()
nb.fit(Xtrain,Ytrain)
Ypred = nb.predict(Xtest)

print("\nNaive Bayes Performance")
print_score(Ytest,Ypred)

"""### SVM"""
svm = sklearn.svm.SVC()
svm.fit(Xtrain,Ytrain)
Ypred = svm.predict(Xtest)

print("\nSVM performance")
print_score(Ytest,Ypred)

"""### Neural Network"""
nn = sklearn.neural_network.MLPClassifier()
nn.fit(Xtrain,Ytrain)
Ypred = nn.predict(Xtest)

print("\nNeural Network Performance")
print_score(Ytest,Ypred)

"""### KNN"""
knn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
knn5 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
knn5d = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,weights='distance')

knn1.fit(Xtrain,Ytrain)
knn5.fit(Xtrain,Ytrain)
knn5d.fit(Xtrain,Ytrain)

print("\nKNN 1 Neighbor Performance")
print_score(Ytest,knn1.predict(Xtest))

print("\nKNN 5 Neighbor Performance")
print_score(Ytest,knn5.predict(Xtest))

print("\nKNN 5 Neighbor Weighted Performance")
print_score(Ytest,knn5d.predict(Xtest))
