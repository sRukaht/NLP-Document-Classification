import numpy as np
import os
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from collections import namedtuple
from random import shuffle
from math import floor
import pandas as pd
from sklearn import metrics
import pickle

def get_tag_and_training_data():
tags=[]
documents=[]
data = pd.read_csv("labeledDataset_8Classes.csv")
for row, datum in data.iterrows():

try:
file = open("ALLtextFilesPROCESSED/"+datum['id'])
FILE = file.read().replace('\n', ' ')
documents.append(FILE)
tags.append(datum['type'])
except:
pass

return tags,documents


Y,X=get_tag_and_training_data()
labels = list(set(Y))
print(labels)

count = len((X))
count = int(0.65 * count)
Y_train,Y_test=Y[:count],Y[count:]
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X)
freq_term_matrix = count_vectorizer.transform(X)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.transform(freq_term_matrix)


#train logistic regression model
X_train,X_test=tf_idf_matrix[:count],tf_idf_matrix[count:]
logreg = linear_model.LogisticRegression(C=1000000)
logreg.fit(X_train,Y_train)
pred=logreg.predict(X_test)

#Save the trained model
saved_model = pickle.dumps(logreg)
#load the pickled model
logreg_from_pickle = pickle.loads(saved_model)

logreg_from_pickle.predict(X_test)

B = accuracy_score(Y_test, pred)
print("The accuracy score is: " + str(B))

E = precision_score(Y_test, pred, average="micro")
print("The precision score is: " + str(E))

F = recall_score(Y_test, pred,average="micro")
print("The recall score is: " +str(F))

D = f1_score(Y_test,pred, average="micro")
print("The f1 score is: " + str(D))


#To plot the confusion matrix
cnf_matrix = metrics.confusion_matrix(Y_test, pred, labels)
print(cnf_matrix)

