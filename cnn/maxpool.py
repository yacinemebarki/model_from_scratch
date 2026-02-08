import numpy as np
class maxpool:
    def __init__(self,pool_size,stirde):
        self.pool_size=pool_size
        self.stride=stirde
        self.type="maxpool"
        

    def forward(self,input):
        
        
        if input.ndim==2:
            input=input[...,np.newaxis]
        self.input=input    
        c=self.input.shape[2] 
        H=(self.input.shape[0]-self.pool_size[0])//self.stride +1
        W=(self.input.shape[1]-self.pool_size[1])//self.stride +1   
        out=np.zeros((H,W,c))    
        for s in range(c):
            for i in range(H):
                for j in range(W):
                    patch=self.input[i*self.stride:i*self.stride+self.pool_size[0],j*self.stride:j*self.stride+self.pool_size[1],s] 
                    out[i,j,s]=np.max(patch)
        self.output=np.array(out)                
        return np.array(out)  
    def backdrop(self,dout,lr):
            
        dx = np.zeros_like(self.input)
        H, W, c = dout.shape 
        for s in range(c):
            for i in range(H):
                for j in range(W):
                    patch=self.input[i*self.stride:i*self.stride+self.pool_size[0],j*self.stride:j*self.stride+self.pool_size[1],s]
                    ind=np.unravel_index(np.argmax(patch),patch.shape)
                    dx[i*self.stride+ind[0],j*self.stride+ind[1],s]+=dout[i,j,s]
                   
        return dx            

                                      
            
