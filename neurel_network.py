import numpy as np
from softmax_regressor import softmax
from my_models import sigmoid
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

    for i in range(epochs):

        loss=0
        for t in range(n_samples):
            for j in range(n_layer):
                if i==0:
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
                else:
                
                    
                    z=a_prev @ w +b
                    a=sigmoid(z)
                    a_prev=a
                    A.append(a)
                    
               
                
            z=a_prev @ wights[-1] + biases[-1]
            p=softmax(z)
            dZ_out = p-y_onehot[t]
            dW_out = a_prev.T @ dZ_out
            db_out = np.sum(dZ_out, axis=0)
            loss += -np.log(p[y[t]])
            for l in reversed(range(n_layer)):
                da=dZ_out[l+1]@ wights[l+1].T
                dZ_out[l]=da * sigmoid_derivative(z)
                dw[l] = np.outer(A[l], dZ_out[l])
                dp[l] = dZ_out[l]

            
            for k in range(len(wights)) :
                wights[k]-=learning_rate*dw[k]
                biases[k]-=learning_rate*dp[k]
    return wights,biases
        
                      

                    



    