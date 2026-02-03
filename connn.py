import numpy as np
from neurel_network import sigmoid,sigmoid_derivative,softmax
class kernel:
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
class layer:
    def __init__(self):
        self.layers=[]
    class conv:
        def __init__(self,n_kernel,kernel_size,stride,input_shape):
            self.n_kernel=n_kernel
            self.kernel_size=kernel_size
            self.stride=stride
            self.input_shape=input_shape
            self.kernels=[]
            self.type="conv"
            for _ in range(n_kernel):
                weight=np.random.randn(kernel_size,kernel_size)*0.01
                bias=0
                kern=kernel(weight,bias)
                self.kernels.append(kern)
    def add_conv(self,n_kernel,kernel_size,stride,input_shape):
        conv_layer=self.conv(n_kernel,kernel_size,stride,input_shape)
        self.layers.append(conv_layer)            

    class flatten:
        def __init__(self,n_neurons,weight,bias):
            self.weight=None
            self.bias=None
            self.n_neurons=n_neurons
            self.type="flatten"
    def add_flatten(self):
        flatten_layer=self.flatten()
        self.layers.append(flatten_layer)
    class maxpool:
        def __init__(self,pool_size,stride,weight,bias):
            self.weight=weight
            self.bias=bias
            self.pool_size=pool_size
            self.stride=stride
            self.type="maxpool"
    def add_maxpool(self,pool_size,stride):
        maxpool_layer=self.maxpool(pool_size,stride)
        self.layers.append(maxpool_layer)            


def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def kernel_conv(data,kern,stride):
    out=[]
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            patch=data[i*stride:i*stride+kern.weight.shape[0],j*stride:j*stride+kern.weight.shape[1]]
            conv=np.sum(patch * kern.weight) + kern.bias
            out.append(relu(conv))
    return  np.array(out).reshape(( (data.shape[0]-kern.weight.shape[0])//stride +1 , (data.shape[1]-kern.weight.shape[1])//stride +1 ))        
                
                


def cnn(x,y,model,learning_rate=0.01,epochs=1000):
    x=np.array(x)
    y=np.array(y)
    x = np.expand_dims(x, axis=-1)
    n_samples,n_hight,n_width,channels=x.shape
    n_class=len(np.unique(y))
    y_onehot=np.eye(n_class)[y]
    flatten_weights=[]
    flatten_biases=[]
    n_flatten_layers=0
   
    for epoch in range(epochs):
        loss=0
        for t in range(n_samples):
            A=[]
            Z=[]
            Aconv=[]
            a=x[t]
            for l in model.layers:
                if l.type=="conv":
                    out=[]
                    Aconv.append(a.copy())
                    for kern in l.kernels:
                        out.append(kernel_conv(a,kern,l.stride))
                    a=np.array(out)
                    
                        
                elif l.type=="maxpool":
                    if len(a.shape)==2:
                        h=(a.shape[0] - l.pool_size[0]) // l.stride + 1
                        w=(a.shape[1] - l.pool_size[1]) // l.stride +1
                        output=np.zeros((h,w))
                        for i in range(0,a.shape[0]):
                            for j in range(0,a.shape[1]):
                                patch=a[i*l.stride:i*l.stride+l.pool_size.shape[0],j*l.stride:j*l.stride+l.pool_size.shape[1]]
                                output[i,j]=np.max(patch)
                        a=output
                    elif len(a.shape)==3:
                        for d in range(a.shape[0]):
                            for i in range(0,a.shape[1]):
                                for j in range(0,a.shape[2]):
                                    patch=a[d,i*l.stride:i*l.stride+l.pool_size.shape[0],j*l.stride:j*l.stride+l.pool_size.shape[1]]
                                    output[d,i,j]=np.max(patch)
                        a=output            

                elif l.type=="flatten":
                    a=a.reshape(-1)
                    if epoch==0 and t==0:
                        n_flatten_layers+=1

                        
                        weight=np.random.randn(a.shape[0]*a.shape[1],l.n_neurons)*0.01
                        bias=np.zeros(l.n_neurons)
                        l.weight=weight
                        l.bias=bias
                        flatten_weights.append(weight)
                        flatten_biases.append(bias)
                    

                    z=a@l.weight+l.bias
                    a=sigmoid(z)
                    A.append(a)
                    Z.append(z)
            if epoch==0 and t==0:
                w_out=np.random.randn(l.n_neurons,n_class)*0.01
                b_out=np.zeros(n_class)
                flatten_weights.append(w_out)
                flatten_biases.append(b_out)
            z_out=a @ w_out + b_out
            a_out=softmax(z_out)
            
            loss+=-np.sum(y_onehot[t]*np.log(a_out+1e-8))
            d_out=a_out - y_onehot[t]
            dw_out=np.outer(A[-1],d_out)
            db_out=d_out
            
            dA_prev=d_out @ w_out.T
            dw=[0]*n_flatten_layers
            db=[0]*n_flatten_layers
            idx=0
            idx = 0
            for j in reversed(range(len(model.layers))):
                if model.layers[j].type == "flatten":
                    dZ = dA_prev * sigmoid_derivative(Z[j])
                if j == 0:
                    dw[j] = np.outer(a, dZ)
                else:
                    dw[j] = np.outer(A[j-1], dZ)
                db[j] = dZ
                dA_prev = dZ @ model.layers[j].weight.T
                if model.layers[j].type == "conv":
                    a_input = Aconv[idx]  
                    for k_idx, kern in enumerate(model.layers[j].kernels):
                        H_out = (a_input.shape[0] - kern.weight.shape[0]) // model.layers[j].stride + 1
                        W_out = (a_input.shape[1] - kern.weight.shape[1]) // model.layers[j].stride + 1
                        grad = dA_prev[k_idx] if len(dA_prev.shape) == 2 else np.ones((H_out, W_out))
                        dK = np.zeros_like(kern.weight)
                        db = 0
                        for i in range(H_out):
                            for c in range(W_out):
                                patch = a_input[i*model.layers[j].stride:i*model.layers[j].stride+kern.weight.shape[0],
                                            c*model.layers[j].stride:c*model.layers[j].stride+kern.weight.shape[1]]
                                z = np.sum(patch * kern.weight) + kern.bias
                                dZ_conv = grad[i, c] * relu_derivative(z)
                                dK += patch * dZ_conv
                                db += dZ_conv

                    kern.weight -= learning_rate * dK
                    kern.bias -= learning_rate * db
                    idx += 1


            for j in range(len(model.layers)):
                if model.layers[j].type=="flatten":
                    model.layers[j].weight -= learning_rate * dw[j]
                    model.layers[j].bias -= learning_rate * db[j]
                
            w_out -= learning_rate * dw_out
            b_out -= learning_rate * db_out
    return model,w_out,b_out  
def cnn_predict(x,model,w_out,b_out):
    x=np.array(x)
    x = np.expand_dims(x, axis=-1)
    n_samples,n_hight,n_width,channels=x.shape
    n_class=w_out.shape[1]
    predictions=[]
    for t in range(n_samples):
        a=x[t]
        for l in model.layers:
            if l.type=="conv":
                out=[]
                for kern in l.kernels:
                    out.append(kernel_conv(a,kern,l.stride))
                a=np.array(out)
            elif l.type=="maxpool":
                if len(a.shape)==2:
                    h=(a.shape[0] - l.pool_size[0]) // l.stride + 1
                    w=(a.shape[1] - l.pool_size[1]) // l.stride +1
                    output=np.zeros((h,w))
                    for i in range(0,a.shape[0]):
                        for j in range(0,a.shape[1]):
                            patch=a[i*l.stride:i*l.stride+l.pool_size.shape[0],j*l.stride:j*l.stride+l.pool_size.shape[1]]
                            output[i,j]=np.max(patch)
                    a=output
                elif len(a.shape)==3:
                    for d in range(a.shape[0]):
                        for i in range(0,a.shape[1]):
                            for j in range(0,a.shape[2]):
                                patch=a[d,i*l.stride:i*l.stride+l.pool_size.shape[0],j*l.stride:j*l.stride+l.pool_size.shape[1]]
                                output[d,i,j]=np.max(patch)
                    a=output            
            elif l.type=="flatten":
                a=a.reshape(-1)
                a=sigmoid(a @ l.weight + l.bias)
        z_out=a @ w_out + b_out
        a_out=softmax(z_out)
        predictions.append(np.argmax(a_out))
    return np.array(predictions)

            

            

    
                        
