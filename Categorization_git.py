# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 08:57:27 2016

@author: rezeh001c
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup 
import pyodbc
#from nltk.stem.snowball import SnowballStemmer
import wordsegment
from wordsegment import segment  
import random
from nltk.corpus import stopwords
import random
from sklearn.cross_validation import train_test_split
from nltk.stem.lancaster import LancasterStemmer  
st = LancasterStemmer()  



# For simplicity, read data from an Excel file with multiple worksheets 
readtrain=pd.read_excel('QA.xlsx', sheetname='Train', header=0, skiprows=None, \
index_col=None, parse_cols="A,B,C,D,E")
readtrain.columns # Display column names of the file 
cnt=len(readtrain)


def letter(w):
    if w.lower() not in lines:
        i=re.sub("[^a-zA-Z]", " ", w)
    else:
        i=w
    return i

def seg(w):
    if w.lower() in lines:
        r = w.lower()
    else:
        r= segment(w)
    return r
def idea_to_words( raw_idea ):
    # Function to convert a raw ideas to a string of words
    # The input is a single string and the output is a single string 
    # (a preprocessed idea text) 
    # 1. Remove HTML
    idea_text = BeautifulSoup(raw_idea, "lxml").get_text() 
    # Use regular expressions to do a find-and-replace to remove non-letters and punctuations  
    letters_only = re.sub("[^a-zA-Z(x1)']", " ", idea_text) 
    #
    #  Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # Remove stop words, split lumped words and put the tokenized words back together.
    meaningful_words = [" ".join(seg(w)) for w in words if not w in stops] 
    # Join the words back into one string separated by space, and return the result.
    return( " ".join( meaningful_words )) 

             
num_readtrain=len(readtrain)
trainfile=[]
catfile =[]
for i in xrange( 0, num_readtrain ):
    # Call our function for each one, and add the result to the list of clean ideas
    # If the index is evenly divisible by 100, print a message
    if( (i+1)%100 == 0 ):
        print(" Retrieving Idea %d of %d\n" % ( i+1, num_readtrain ))     
    trainfile.append( idea_to_words(readtrain['TITLE'][i])+ ' '+idea_to_words(readtrain['DESCRIPTION'][i]) )
    catfile.append([readtrain['CATEGORY'][i],readtrain['TG2'][i]])


exam=pd.read_excel('QA.xlsx', sheetname='examine', header=0, skiprows=None, \
index_col=None, parse_cols="A,B,C,D,E,F")
exam.columns # Display column names if the file 
cnt=len(exam)

num_eval=len(exam)
evalfile=[]
for i in xrange( 0, num_eval ):
    if( (i+1)%100 == 0 ):
        print(" Retrieving Idea %d of %d\n" % ( i+1, num_eval ))     
    evalfile.append (idea_to_words(exam['TITLE'][i])+ ' '+idea_to_words(exam['DESCRIPTION'][i]))



from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trainfile)
X_train_counts.shape # (2544, 14309)


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = X_train_counts = count_vect.fit_transform(trainfile)
X_train_counts.shape ## (2544, 14309)


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape # (2544, 14309)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape # (2544, 14309)

Train_target = readtrain['TG2'].as_matrix()
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, Train_target)


from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),])
text_clf = text_clf.fit(trainfile, Train_target)

Test_target = exam['TG '].as_matrix()
Idea_cat = evalfile
docs_test = Idea_cat
#predicted = text_clf.predict(docs_test)
#np.mean(predicted == Test_target) # 0.80000000000000004

'''I have 80% success rate. Let’s see if I can do better with a linear support vector machine (SVM)'''

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='modified_huber', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(trainfile, Train_target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == exam['TG '])   ## 0.86829268292682926



def convert_to_name(cat_id):
    if cat_id == 1:
        catname = 'Technology'
    elif cat_id == 2:
        catname = 'Customer Experience'
    elif cat_id == 3:
        catname = 'New Business'
    else:
        catname = 'Process'
    return catname
    


def pred(idea):
    f=[]
    f.append(idea_to_words(idea))
    predicted = text_clf.predict(f)
    return convert_to_name(predicted)      

def main():
    idea = raw_input("Please enter an idea: ")
    return pred(idea)
    
if __name__ == '__main__':
     main()
     