from model import linear
import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from TFIDf import compute_tf


X = np.random.randn(30000,1)
y = np.random.randn(30000)
model1=LinearRegression()
model2=linear()
start=time.time()
model1.fit(X,y)
end=time.time()
print("sklearn time: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)

start=time.time()
model2.fit(X,y)
end=time.time()
print("custom model time: ",end-start)
text=["are you learning machine learning",
      "machine learning is fun",
      "I love coding in python",
      "python is a great programming language",
      "do you love deep learning",
      "i hate bugs in my code",
      "i hate information systems"]
labal=[0,1,1,1,0,0,0]
tfidf_sklearn=TfidfVectorizer()
start=time.time()
X_sklearn=tfidf_sklearn.fit_transform(text).toarray()
end=time.time()
print("sklearn TFIDF time: ",end-start)
print("sklearn TFIDF result:\n",X_sklearn)
start=time.time()
X_custom, vocab=compute_tf(text)
end=time.time()
print("custom TFIDF time: ",end-start)
print("custom TFIDF result:\n",np.array(X_custom))
start=time.time()
model1.fit(X_sklearn,labal)
end=time.time()
print("sklearn model time with TFIDF: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)
start=time.time()
model2.fit(X_custom,labal)
end=time.time()
print("custom model time with TFIDF: ",end-start)
print("model coefficients: ",model2.A)
print("model intercept: ",model2.B)
