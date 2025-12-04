from abc import ABC, abstractmethod
import math
import numpy as np

class model(ABC):
    @abstractmethod
    def fit(self,x,y):
        pass
    @abstractmethod
    def predict(self,x):
        pass
    
class linear(model):
    
    def __init__(self,A=0,B=0):
        self.A=A
        self.B=B
    def fit(self,x,y):
        n=len(x)
        if n==0 or n!=len(y):
            print("you cant enter empty array")
            return
        else :
            s1=0
            s2=0
            s3=0
            s=0
            for i in range(n):
                s=s+x[i]*y[i]
                s1=s1+x[i]
                s2=s2+y[i]
                s3=s3+x[i]**2
            A=(n*s-(s2*s1))/(n*s3-s1**2)
            A=np.array(A)
            A=np.round(A,4)
            B=(s2 - A*s1)/n
            self.A=A
            self.B=B
            print("the slop is: ",A," the intercept: ",B)   
    def predict(self,x):
        y=[]
        for i in range(len(x)):
            t=self.A*x[i]+self.B
            y.append(t)
        return y
    def MSE(self,y,y2):
        n=len(y)
        if n==0 or n!=len(y2):
            print("empty array")
            return
        else :
            s=0
            for i in range(n):
                s=s+(y[i]-y2[i])**2
            return s/n
    def RMSE(self,y,y2):
        t=self.MSE(y,y2)
        return math.sqrt(t)
    def MEA(self,y,y2):
        n=len(y)
        if n==0 or n!=len(y2):
            print("empty array")
            return
        else :
            s=0
            for i in range(n):
                s=s+abs(y[i]-y2[i])
            return s/n    
               
                



               
