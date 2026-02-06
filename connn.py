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
        def __init__(self,n_kernel,kernel_size,stride,input_shape,input):
            self.n_kernel=n_kernel
            self.kernel_size=kernel_size
            self.stride=stride
            self.input_shape=input_shape
            self.kernels=[]
            self.input=input
            self.type="conv"
            for _ in range(n_kernel):
                weight=np.random.randn(kernel_size[0],kernel_size[1])*0.01
                bias=0
                kern=kernel(weight,bias)
                self.kernels.append(kern)
    def add_conv(self,n_kernel,kernel_size,stride,input_shape,input=None):
        conv_layer=self.conv(n_kernel,kernel_size,stride,input_shape,input)
        self.layers.append(conv_layer)            

    class flatten:
        def __init__(self,n_neurons,weight,bias,input):
            self.weight=None
            self.bias=None
            self.n_neurons=n_neurons
            self.type="flatten"
            self.input=input
    def add_flatten(self,n_neurons):
        flatten_layer=self.flatten(n_neurons,None,None,None)
        self.layers.append(flatten_layer)
    class maxpool:
        def __init__(self,pool_size,stride,weight,bias):
            self.weight=weight
            self.bias=bias
            self.pool_size=pool_size
            self.stride=stride
            self.type="maxpool"
    def add_maxpool(self,pool_size,stride):
        maxpool_layer=self.maxpool(pool_size,stride,None,None)
        self.layers.append(maxpool_layer)            


def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def kernel_conv(data,kern,stride):
    if len(data.shape)==3:
        H,W,C=data.shape
    else :
        H,W=data.shape
        C=1
        data=data.reshape(H,W,1)    
    
    H=(data.shape[0]-kern.weight.shape[0])//stride +1
    W=(data.shape[1]-kern.weight.shape[1])//stride +1
    out=np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            patch=data[i*stride:i*stride+kern.weight.shape[0],j*stride:j*stride+kern.weight.shape[1]]
            conv=np.sum(patch * kern.weight[:,:]) + kern.bias
            out[i,j]=relu(conv)
    return  out        
def Maxpool(data,pool_size,stride):
    if len(data.shape)==3:
        H,W,C=data.shape
    else:
        H,W=data.shape
        C=1
        data=data.reshape(H,W,1)
    H=(data.shape[0]-pool_size[0])//stride +1
    W=(data.shape[1]-pool_size[1])//stride +1
    out=np.zeros((H,W,C))
    for  c in range(C):
        for i in range(H):
            for j in range(W):
                patch=data[i*stride:i*stride+pool_size[0],j*stride:j*stride+pool_size[1],c]
                out[i,j,c]=np.max(patch)
    return out      
                


def cnn(x,y,model,learning_rate=0.01,epochs=1000):
    x=np.array(x)
    y=np.array(y)
    if len(x.shape)==3:
        x = np.expand_dims(x, axis=-1)
    n_samples,n_hight,n_width,channels=x.shape
    n_class=len(np.unique(y))
    y_onehot=np.eye(n_class)[y]
    flatten_weights=[]
    flatten_biases=[]
    n_flatten_layers=0
    print(n_samples)
   
    for epoch in range(epochs):
        loss=0
    
        for t in range(n_samples):
           
            Z=[]
            
            a=x[t]
            print("the input shape first ",a.shape[2])
            s=0
            for l in model.layers:
                if l.type=="conv":
                    out=[]
                    l.input=a.copy()
                    print("conv",s)
                    s=s+1
                    for kern in l.kernels:
                        out.append(kernel_conv(a,kern,l.stride))
                    a=np.array(out)
                    print("the outupot of conv layer",a)
                    if t==0 and epoch==0:
                        print(l.kernels[0].weight)
                    
                        
                elif l.type=="maxpool":
                    
                    if len(a.shape)==2:
                        a=Maxpool(a,l.pool_size,l.stride)
                        print("maxpool",s)
                        s=s+1
                        
                       
                    elif len(a.shape)==3:
                        output=np.zeros((a.shape[0],
                                         (a.shape[1] - l.pool_size[0]) // l.stride + 1,
                                         (a.shape[2] - l.pool_size[1]) // l.stride + 1))
                        for d in range(a.shape[0]):
                            for i in range(0,output.shape[1]):
                                for j in range(0,output.shape[2]):
                                    patch=a[d,i*l.stride:i*l.stride+l.pool_size[0],j*l.stride:j*l.stride+l.pool_size[1]]
                                    output[d,i,j]=np.max(patch)
                        a=output 
                        print("maxpool",s)
                        s=s+1           

                elif l.type=="flatten":
                    a=a.reshape(-1)
                    l.input=a.copy()
                    print("flatt",s)
                    s=s+1
                    if epoch==0 and t==0:
                        n_flatten_layers+=1

                        
                        weight=np.random.randn(a.size,l.n_neurons)*0.01
                        bias=np.zeros(l.n_neurons)
                        l.weight=weight
                        l.bias=bias
                        flatten_weights.append(weight)
                        flatten_biases.append(bias)
                    

                    z=a@l.weight+l.bias
                    a=sigmoid(z)
                    Z.append(z)
            if epoch==0 and t==0:
                
                w_out=np.random.randn(l.n_neurons,n_class)*0.01
                print(w_out)
                b_out=np.zeros(n_class)
                flatten_weights.append(w_out)
                flatten_biases.append(b_out)
            z_out=a @ w_out + b_out
            a_out=softmax(z_out)
            
            loss+=-np.sum(y_onehot[t]*np.log(a_out+1e-8))
            d_out=a_out - y_onehot[t]
            fc_input = a.reshape(-1) if model.layers[-1].type=="flatten" else model.layers[-1].input
            dw_out=np.outer(fc_input,d_out)

            db_out=d_out
            
            dA_prev=d_out @ w_out.T
            dw=[0]*n_flatten_layers
            db=[0]*n_flatten_layers
            
            idx_flat=len(Z)-1
            for j in reversed(range(len(model.layers))):
    
                layer_type = model.layers[j].type
                print(layer_type)

    
                

                if layer_type == "flatten":
                    
                    dZ = dA_prev * sigmoid_derivative(Z[idx_flat])
                    dw[idx_flat] = np.outer(model.layers[j].input, dZ)
                    db[idx_flat] = dZ
                    dA_prev = dZ @ model.layers[j].weight.T
                    idx_flat -= 1
                    print(f"flatten layer {j} updated at epoch {epoch}")

                elif layer_type == "conv":
                    a_inp = model.layers[j].input  
                    dA_prev_conv = np.zeros_like(a_inp)

                   
                    for k, kern in enumerate(model.layers[j].kernels):
                        kh, kw = kern.weight.shape
                        H_out = (a_inp.shape[1] - kh) // model.layers[j].stride + 1
                        print("the inputshape",a_inp.shape[2])
                        
                        W_out = (a_inp.shape[2] - kw) // model.layers[j].stride + 1
                        
                        dk = np.zeros_like(kern.weight)
                        dbk = 0
                        print("enter to the kernel")
                        print("hieght",H_out)
                        print("width",W_out)


                        for i in range(H_out):
                            for c in range(W_out):
                                
                                patch = a_inp[k, 
                                              i*model.layers[c].stride:i*model.layers[c].stride+kh,
                                              c*model.layers[c].stride:c*model.layers[c].stride+kw]
                                print("patch",patch)

                                
                                dz = dA_prev[k, i, c] * relu_derivative(np.sum(patch * kern.weight) + kern.bias)
                                print("dz",dz)
                            

                                dk += patch * dz
                                dbk += dz
                                
                                dA_prev_conv[:, 
                                             i*model.layers[t].stride:i*model.layers[t].stride+kh,
                                             t*model.layers[t].stride:t*model.layers[t].stride+kw] += kern.weight * dz
                                

                        
                        kern.weight -= learning_rate * dk
                        kern.bias -= learning_rate * dbk
                        print(dk)
                        print(dbk)

                        
                        dA_prev = dA_prev_conv
                        print(f"conv layer {j} updated at epoch {epoch}")




                
            idx=0
            for j in range(len(model.layers)):
                if model.layers[j].type=="flatten":
                    model.layers[j].weight -= learning_rate * dw[idx]
                    model.layers[j].bias -= learning_rate * db[idx]
                    idx+=1
            w_out -= learning_rate * dw_out
            b_out -= learning_rate * db_out
            print(epoch)
    return model,w_out,b_out  
def cnn_predict(x,model,w_out,b_out):
    x=np.array(x)
    if len(x.shape)==3:
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
                
                a=Maxpool(a,l.pool_size,l.stride)
            elif l.type=="flatten": 
                a=a.reshape(-1)
                z=a @ l.weight + l.bias
                a=sigmoid(z)
        z_out=a @ w_out + b_out
        a_out=softmax(z_out)
        pred=np.argmax(a_out)
        predictions.append(pred)
    return np.array(predictions)
    
            

    
                        
