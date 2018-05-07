# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# quoting = 3 ignores '"'
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords') # stopwords - unwanted words (eg - the, this, so...)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # Keep only a-z and A-Z words and others replaced by ' '
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() # Change word into root word(eg - loved become love).
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# Create matrix with words.Colums - The word present in the Nth review or not 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # max_features limits the filtered words into 1500, columns = 1500 words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
# Based on experience Naivebayes, Random Forest, Decisiomn tree algoritm are best for NLP
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

''' 
Confusion Matrix
    [[55 42]
     [12 91]]
    
    42 - false positives
    12 - false negatives
'''

# Find Accuracy
# 200 test sets
# Correct predictions from cm - 55+91 = 146

accuracy = 146 / 200 # 0.73