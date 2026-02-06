import numpy as np
def relu(input):
    return np.maximum(0,input)
class kernel:
    def __init__(self,size,weight,bias):
        self.weight=weight
        self.bias=bias
        self.size=size
class conv:
    def __init__(self,n_kernel,kernel_size,input_shape,stride,input):
        self.kernels=[]
        self.stride=stride
        self.input=input
        self.input_shape=input_shape
        self.kernel_size=kernel_size
        for i in range(n_kernel):
            ker=kernel(kernel_size,None,None)
            if len(input_shape)==3:
                ker.weight=np.random.rand(kernel_size[0],kernel_size[1],input_shape[2])*0.01
                ker.bias=0.0
            else :
                ker.weight=np.random.rand(kernel_size[0],kernel_size[1])*0.01
                ker.bias=0.0
            self.kernels.append(ker)
    def forward(self):
        output=[]
        H=(self.input_shape[0]-self.kernel_size[0])//self.stride +1
        W=(self.input_shape[1]-self.kernel_size[1])//self.stride +1
        for ker in self.kernels :
            
            
            
            out=np.zeros((H,W))
             
            
            for i in  range(H):
                for j in range(W):
                    patch=self.input[i*self.stride:i*self.stride+self.kernel_size[0],j*self.stride:j*self.stride+self.kernel_size[1],:]
                    conv=np.sum(patch*ker.weight)+ker.bias
                    out[i,j]=relu(conv)
            output.append(out)
        self.output=output
        return output                

        



