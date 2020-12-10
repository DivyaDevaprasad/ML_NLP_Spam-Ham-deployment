import pandas as pd

data = pd.read_csv("spam.csv", encoding="latin-1")

# Features and Labels
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
data.columns = ["class","message"]
data['label'] = data['class'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']
	
# Extract Feature With CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
    

import pickle

pickle.dump(cv, open('tranform.pkl', 'wb'))
    
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)
    
pickle.dump(model, open('nlp_model.pkl', 'wb'))
    
#	Alternative Usage of Saved Model
#	 joblib.dump(clf, 'NB_spam_model.pkl')
#	 NB_spam_model = open('NB_spam_model.pkl','rb')
#	 clf = joblib.load(NB_spam_model)# -*- coding: utf-8 -*-

