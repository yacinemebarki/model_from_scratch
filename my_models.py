from abc import ABC, abstractmethod
import math
import numpy as np
from decision_tree_algorithm import fit_tree
from softmax_regressor import fit_softmax

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
class scaler:
    def __init__(self):
        self.min=None
        self.max=None
        
    def transform(self,x):
        x=np.array(x, dtype=float)
        self.min=np.min(x,axis=0)
        self.max=np.max(x,axis=0)
        if self.min is None or self.max is None:
            print("you need to fit the model first")
            return
        scaled_x=(x - self.min)/(self.max - self.min)
        return scaled_x
    def inverse_transform(self,x):
        x=np.array(x, dtype=float)
        if self.min is None or self.max is None:
            print("you need to fit the model first")
            return
        original_x=x * (self.max - self.min) + self.min
        return original_x

#linear regression and tfidf
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
        
    def scaler(self,x):
        scalerd_x=scaler(x)
        return scalerd_x    
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
#logistic regression    
class logistic(model):

    def __init__(self,n,A=0,B=0,use=None):
        self.A=A
        self.n=n
        self.B=B
        self.use=use
    def scaler(self,x):
        scalered_x=scaler(x)
        return scalered_x    
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
        
    
    def confusion_matrix(self,y_true,y_pred):
        if len(y_true)==0 or len(y_true)!=len(y_pred):
            print("empty array")
            return
        True_positive=0
        False_positive=0
        True_negative=0
        False_negative=0
        for i in range(len(y_true)):
            
            if y_true[i]==1 and y_pred[i]==1:
                True_positive=True_positive+1
            elif y_true[i]==0 and y_pred[i]==1:
                False_positive=False_positive+1
            elif y_true[i]==0 and y_pred[i]==0:
                True_negative=True_negative+1
            elif y_true[i]==1 and y_pred[i]==0:
                False_negative=False_negative+1
        return np.array([[True_positive,False_positive],[False_negative,True_negative]])
    def accuracy(self,y_true,y_pred):
        a=self.confusion_matrix(y_true,y_pred) 
        return (a[0,0]+a[1,1])/np.sum(a)
    def precision(self,y_true,y_pred):
        a=self.confusion_matrix(y_true,y_pred) 
        return a[0,0]/(a[0,0]+a[0,1])
    def recall(self,y_true,y_pred):
        a=self.confusion_matrix(y_true,y_pred) 
        return a[0,0]/(a[0,0]+a[1,0])
    def f1_score(self,y_true,y_pred):
        p=self.precision(y_true,y_pred)
        r=self.recall(y_true,y_pred)
        return 2*(p*r)/(p+r)
class decision_tree(model):
    def __init__(self,root=None):
        self.root=root
    def fit(self,x,y):
        self.root=fit_tree(x,y)
        return self.root
    def print_tree(self):
        from decision_tree_algorithm import print_tree
        print_tree(self.root)
    def predict(self,root,x):
        if root is None:
        
            return
        if len(np.unique(root.lable)) == 1:
            return root.lable[0]
        if x == 0:
            return self.predict(root.left, x)
        else:
            return self.predict(root.right, x)
from decision_tree_algorithm import fit_regression,print_tree_regression        
class decision_tree_regression(model):
    def __init__(self,root=None,max_depth=float('inf'),min_samples=2):
        self.root=root
        self.max_depth=max_depth
        self.min_samples=min_samples
    def fit(self,x,y):
        self.root=fit_regression(x,y,self.max_depth,self.min_samples)
        return self.root
    def print_tree(self):
        
        print_tree_regression(self.root)
    def predict(self,root,x):
        if root is None:
            return
        if root.left is None and root.right is None:
            return root.value
        if x <= root.threshold:
            return self.predict(root.left, x)
        else:
            return self.predict(root.right, x)        
#softmax regression
class softmax_regression(model):
    def __init__(self,weights=None,bias=None):
        self.weights=weights
        self.bias=bias
    def fit(self,x,y,learning_rate=0.01,epochs=1000):
        
        self.weights,self.bias=fit_softmax(x,y,learning_rate,epochs)
        return self.weights,self.bias
    def predict(self,x):
        x=np.array(x)
        n_samples=x.shape[0]
        z=np.dot(x, self.weights) + self.bias
        p=softmax(z)
        predictions=np.argmax(p, axis=1)
        return predictions