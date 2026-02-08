import numpy as np
def relu(input):
    return np.maximum(0,input)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
class kernel:
    def __init__(self,size,weight,bias):
        self.weight=weight
        self.bias=bias
        self.size=size
class conv:
    def __init__(self,n_kernel,kernel_size,input_shape,stride):
        self.kernels=[]
        self.stride=stride
        
        self.input_shape=input_shape
        self.kernel_size=kernel_size
        self.type="conv"
        for i in range(n_kernel):
            ker=kernel(kernel_size,None,None)
            if len(input_shape)==3:
                ker.weight=np.random.rand(kernel_size[0],kernel_size[1],input_shape[2])*0.01
                ker.bias=0.0
            else :
                ker.weight=np.random.rand(kernel_size[0],kernel_size[1])*0.01
                ker.bias=0.0
            self.kernels.append(ker)
    def forward(self,input):
        output=[]
        self.input=input
        H=(self.input_shape[0]-self.kernel_size[0])//self.stride +1
        W=(self.input_shape[1]-self.kernel_size[1])//self.stride +1
        self.z=[]
        
        for ker in self.kernels :
            
            
            
            out=np.zeros((H,W))
            z_map=np.zeros((H,W)) 
            
            for i in  range(H):
                for j in range(W):
                    patch=self.input[i*self.stride:i*self.stride+self.kernel_size[0],j*self.stride:j*self.stride+self.kernel_size[1],:]
                    conv=np.sum(patch*ker.weight)+ker.bias
                    z_map[i,j]=conv
                    out[i,j]=relu(conv)
            output.append(out)
            self.z.append(z_map)
        self.output=np.array(output)
        return np.array(output) 
    def backdrop(self,dout,lr):
        H=(self.input_shape[0]-self.kernel_size[0])//self.stride +1
        W=(self.input_shape[1]-self.kernel_size[1])//self.stride +1
        dX = np.zeros_like(self.input)
        for k,ker in enumerate(self.kernels):
            dw = np.zeros_like(ker.weight)
            db = 0
        
            for i in range(H):
                for j in range(W):
                    patch=self.input[i*self.stride:i*self.stride+self.kernel_size[0],j*self.stride:j*self.stride+self.kernel_size[1],:]
                    dconv=dout[k,i,j]*relu_derivative(self.z[k][i,j])
                    dw+=patch*dconv
                    db+=dout[k,i,j]
                    dX[i*self.stride:i*self.stride+self.kernel_size[0],j*self.stride:j*self.stride+self.kernel_size[1],:] += ker.weight * dconv
            ker.weight-=lr*dw
            ker.bias-=lr*db
        return dX    




    


        



