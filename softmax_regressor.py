import numpy as np
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)
def fit_softmax(x,y,learning_rate=0.01,epochs=1000):
    x=np.array(x)
    y=np.array(y)
    n_samples,n_features=x.shape
    n_classes=len(np.unique(y))
    b=np.zeros(n_classes)
    w=np.random.randn(n_features,n_classes)*0.01
    Y=np.zeros((n_samples,n_classes))
    z=[]
    p=[]
    loss=0
    for i in range(epochs):
        loss=0
        for j in range(n_samples):
            
            z=np.array([x[j] @ w[:,k] + b[k] for k in range(n_classes)])
            p=softmax(z)

            
            loss += -np.log(p[y[j]])
            for k in range(n_classes):
                if k == y[j]:
                    p[k] -= 1
                w[:,k] -= learning_rate * x[j] * p[k]
                b[k] -= learning_rate * p[k]
    return w,b            