import numpy as np
from softmax_regressor import softmax
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def neural_network(x,y,learning_rate=0.01,n_layer=2,n_neurons=[5,5],epochs=1000):
    x=np.array(x)
    y=np.array(y)
    A=[]
    n_samples,n_features=x.shape
    n_classes=len(np.unique(y))
    wights=[]
    biases=[]
    y=np.array(y)
    y_onehot=np.eye(n_classes)[y]
    dw=[]
    dp=[]
    Z=[]

    for i in range(epochs):

        loss=0
        
        for t in range(n_samples):
            A=[]
            Z=[]
            for j in range(n_layer):
                if i==0 and t==0:
                    if j!=0:
                        n_features=n_neurons[j-1]
                    w=np.random.randn(n_features,n_neurons[j])*0.01
                    wights.append(w)
                    b=np.zeros(n_neurons[j])
                    biases.append(b)
                else:
                    w=wights[j]
                    b=biases[j]
                if j==0:
                    a_prev=x[t]
                    
                    z=a_prev @ w +b
                    a=sigmoid(z)
                    a_prev=a
                    Z.append(z)
                    A.append(a)

                else:
                
                    
                    z=a_prev @ w +b
                    a=sigmoid(z)
                    a_prev=a
                    Z.append(z)
                    A.append(a)
            if i==0 and t==0:
                w_out=np.random.randn(n_neurons[-1],n_classes)*0.01
                b_out=np.zeros(n_classes)
                       
                    
               
                
            z=a_prev @ w_out + b_out
            p=softmax(z)
            
            dZ_out = p-y_onehot[t]
            dw_out=np.outer(A[-1], dZ_out)
            db_out=dZ_out
            dA = dZ_out @ w_out.T

            dw = [0]*n_layer
            db = [0]*n_layer

            
            for l in reversed(range(n_layer)):
                dZ = dA * sigmoid_derivative(Z[l])

                if l==0:
                    dw[l] = np.outer(x[t], dZ)
                else:
                    dw[l] = np.outer(A[l-1], dZ)

                db[l] = dZ

                dA = dZ @ wights[l].T
            
            for k in range(n_layer): 
                wights[k]-=learning_rate*dw[k]
                biases[k]-=learning_rate*db[k]
            w_out-=learning_rate*dw_out
            b_out-=learning_rate*db_out    
    return wights,biases,w_out,b_out
        
                      

                    



    