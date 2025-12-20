import my_models
import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from TFIDf import TFIDF
print(my_models.__file__)



print(hasattr(my_models, "linear"))    
print(hasattr(my_models, "logistic"))  
// testing linear regression model
X = np.random.randn(30000,1)
y = np.random.randn(30000)
model1=LinearRegression()
model2=my_models.linear()
start=time.time()
model1.fit(X,y)
end=time.time()
print("sklearn time: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)
// testing tfidf model
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
tfidf_custom=TFIDF()
X_custom=tfidf_custom.compute_tf(text)
end=time.time()
print("custom TFIDF time: ",end-start)
print("custom TFIDF result:\n",(X_custom))
start=time.time()
model1.fit(X_sklearn,labal)
end=time.time()
print("sklearn model time with TFIDF: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)

start=time.time()
model2.tfidf_fit(X_custom,labal)
end=time.time()
print("custom model time with TFIDF: ",end-start)
print("model coefficients: ",model2.A)
print("model intercept: ",model2.B)
// testing logistic regression model
model3=my_models.logistic(1000)
start=time.time()
model3.fit(X_custom,labal,"softmax")
end=time.time()
print("custom logistic model time with TFIDF: ",end-start)
print("model coefficients: ",model3.A)
print("model intercept: ",model3.B)
model4=LogisticRegression()
start=time.time()
model4.fit(X_sklearn,labal)
end=time.time()
print("sklearn logistic model time with TFIDF: ",end-start)
print("model coefficients: ",model4.coef_)
print("model intercept: ",model4.intercept_)
y_pred=model3.predict(X_custom)
print(y_pred)
y_pred=model4.predict_proba(X_sklearn)
print(y_pred)
text_pr=["i love machine learning and coding in python","python is fun"]
X_sklearn_pr=tfidf_sklearn.transform(text_pr).toarray()
print("sklearn TFIDF transform result:\n",X_sklearn_pr)
X_custom_pr=tfidf_custom.transform(text_pr)
print("custom TFIDF transform result:\n",X_custom_pr)
y_pred_pr=model3.predict(X_custom_pr)
print(y_pred_pr)
y_pred_pr=model4.predict(X_sklearn_pr)
print(y_pred_pr)

