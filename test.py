from model import linear
import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression
import time


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
