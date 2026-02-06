import numpy as np
class maxpool:
    def __init__(self,pool_size,stirde):
        self.pool_size=pool_size
        self.stride=stirde

    def forward(self,input):
        H=(input.shape[0]-self.pool_size[0])//self.stride +1
        W=(input.shape[1]-self.pool_size[1])//self.stride +1
        if len(input.shape)==3:
            c=input.shape[2]
        else :
            input = input.reshape(input.shape[0], input.shape[1], 1)
            c=1
        out=np.zeros((H,W,c))    
        for s in range(c):
            for i in range(H):
                for j in range(W):
                    patch=input[i*self.stride:i*self.stride+self.pool_size[0],j*self.stride:j*self.stride+self.pool_size[1],s] 
                    out[i,j,s]=max(patch)
        self.output=out                
        return out                         
            
