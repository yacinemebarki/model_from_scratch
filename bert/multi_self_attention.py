import numpy as np
from mln import mask

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


class msa:
    def __init__(self,input_size,num_heads):
        self.w=None
        self.q=None
        self.v=None
        dk=input_size//num_heads
        self.dk=dk
        self.input_size=input_size
        self.kw=np.random.rand(input_size,input_size)*0.01
        self.vw=np.random.rand(input_size,input_size)*0.01
        self.qw=np.random.rand(input_size,input_size
                               )*0.01
    
    def forward(self,x):
        x=np.array(x)
        
        dk=self.dk
        
        
        self.x=x
        
        q=x@self.qw
        k=x@self.kw
        v=x@self.vw
        
        score=(q@k.T)/np.sqrt(dk)
        
        weight=softmax(score)
        
        self.q=q
        self.v=v
        self.k=k
        
        self.weight=weight
        out=weight@v
        return out
    
    def backdrop(self,dout,lr):
        dv=self.weight.T@dout
        
        dwv=self.x.T@dv
        
        dweight=dout@self.v.T
        
        dscore=self.weight*(dweight-np.sum(dweight*self.weight,axis=1,keepdims=True))
        
        dq=dscore@self.k/np.sqrt(self.dk)
        dk=dscore@self.q/np.sqrt(self.dk)
        
        dwq=self.x.T@dq
        dwk=self.x.T@dk
        
        self.qw -= lr * dwq
        self.kw -= lr * dwk
        self.vw -= lr * dwv

        dx = dq @ self.qw.T + dk @ self.kw.T + dv @ self.vw.T

        return dx
               
            
            


                                    
                
            
            
        
            