from abc import ABC, abstractmethod
import math
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

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
    def tfidf_fit(self,x,y):
        x=np.array(x, dtype=float)  
        y=np.array(y, dtype=float)
        print(x.shape)
        
        X_bias=np.c_[np.ones(x.shape[0]), x]
        w_all=np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        self.B=w_all[0]
        self.A=w_all[1:]
        return self.A,self.B
class logistic(model):

    def __init__(self,n,A=0,B=0,use=None):
        self.A=A
        self.n=n
        self.B=B
        self.use=use
        
    def fit(self,x,y,use):
        x=np.array(x, dtype=float)
        y=np.array(y, dtype=float)
        x_bias=np.c_[np.ones(x.shape[0]), x]
        w=np.zeros((x_bias.shape[1], 1))
    
        for i in range(self.n):
            z=x_bias @ w
            if use=="sigmoid":
                h=sigmoid(z)
            else :
                h=softmax(z)    
            gradient=x_bias.T @ (h - y.reshape(-1, 1)) / y.size
            w=w-0.01 * gradient
        self.A=w
        self.B=w[0]
        self.use=use
        return self.A,self.B
    def predict_proba(self, x):
        x = np.array(x, dtype=float)
        x_bias = np.c_[np.ones(x.shape[0]), x]
        z = x_bias @ self.A
        if self.use == "sigmoid":
            probs = sigmoid(z)
        else:
            probs = softmax(z)
        return probs    
    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)
        if self.use=="sigmoid":
            return (probs >= threshold).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)
        
    def accuracy(self,y_true,y_pred):
        if len(y_true)==0 or len(y_true)!=len(y_pred):
            print("empty array")
            return
        else:
            correct=0
            for i in range (len(y_true)):
                if y_true[i]==y_pred[i]:
                    correct+=1
            return correct/len(y_true)







          
 





              
                



               
