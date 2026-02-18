import numpy as np
from mln import mask

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


class msa:
    def __init__(self,input_size,num_heads,vocab):
        self.w=None
        self.q=None
        self.v=None
        dk=input_size//num_heads
        self.kw=np.random.rand(input_size,dk)*0.01
        self.vw=np.random.rand(input_size,dk)*0.01
        self.qw=np.random.rand(input_size,dk)*0.01
    def forward(self,x,mask_vec,vocab):
        x=np.array(x)
        
        masked,target=mask(x,vocab,mask_vec)
        
        
        dk=self.kw.shape[1]
        out=[]
        
        for i in range(len(x)):
            tok=masked[i]
            q=tok @ self.qw
            z=np.zeros(dk)
            scores = np.array([q @ (masked[j] @ self.kw) / np.sqrt(dk) for j in range(len(x))])
            weights = softmax(scores)
            for j in range(len(x)):
                
                
                v=masked[j]@self.vw
                z=z+weights[j]*v
            out.append(z)
        return np.array(out),target                        
                
            
            
        
            