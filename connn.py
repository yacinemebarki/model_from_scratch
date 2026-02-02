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
        def __init__(self,weight,bias):
            self.weight=weight
            self.bias=bias
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


def kernel_conv(data,kern,stride):
    for i,j in data:
        patch=data[i:i+stride,j:j+stride]
        z=np.sum(patch*kern.weight)+kern.bias
        out=relu(z)
    return out    

def cnn(x,y,model,learning_rate=0.01,epochs=1000):
    x=np.array(x)
    y=np.array(y)
    x = np.expand_dims(x, axis=-1)
    n_samples,n_hight,n_width,channels=x.shape
    n_class=len(np.unique(y))
    y_onehot=np.eye(n_class)[y]
    
    for epoch in range(epochs):
        loss=0
        for t in range(n_samples):
            a=x[t]
            for l in model.layers:
                if l.type=="conv":
                    for kern in l.kernels:
                        a=kernel_conv(a,kern,l.stride)
                elif l.type=="maxpool":
                    h=(a.shape[0] - l.pool_size[0]) // l.stride + 1
                    w=(a.shape[1] - l.pool_size[1]) // l.stride
                    output=np.zeros((h,w))
                    
                    for i in range(0,a.shape[0],l.stride):
                        for j in range(0,a.shape[1],l.stride):
                            patch=a[i:i+l.pool_size.shape[0],j:j+l.pool_size.shape[1]]
                            output[i,j]=np.max(patch)
                    a=output
                elif l.type=="flatten":
                    if epoch==0:
                        
                        weight=np.random.randn(a.shape[0]*a.shape[1],n_class)*0.01
                        bias=np.zeros(n_class)
                        l.weight=weight
                        l.bias=bias 