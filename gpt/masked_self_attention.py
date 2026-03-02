import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 


class mssa:
    
    
    def __init__(self,input_size,dk,n_token):
        self.q=None
        self.v=None
        self.k=None 
        
        self.input_size=input_size
        self.dk=dk
        self.n_token=n_token
        self.weight=None
        self.vw=np.random.rand(input_size,input_size)*0.01
        self.qw=np.random.rand(input_size,input_size)*0.01
        self.kw=np.random.rand(input_size,input_size)*0.01
    
    def forward(self,x):
        self.x=x
        self.q=x@self.qw
        self.k=x@self.kw
        self.v=x@self.vw
        
        score=self.q@self.k.T/np.sqrt(self.dk)
        
        self.mask=np.triu(np.ones_like(score), k=1).astype(bool)
        score[self.mask]=-1e9
        
        self.weight=softmax(score)
        
        out=self.weight*self.v
        
        return out
    def backdrop(self,dout,lr):
        dv=self.weight.T@dout
        dwv=self.x.T@dv
        
        
        dweight=dout@self.v.T
        dscore=self.weight*(dweight-np.sum(dweight*self.weight,axis=1,keepdims=True))
        dscore[self.mask] = 0
        
        
        dq=dscore@self.k/np.sqrt(self.dk)
        dk=dscore@self.q/np.sqrt(self.dk)
        
        dwq=self.x.T@dq
        dwk=self.x.T@dk
        
        clip = 1.0
        dwq = np.clip(dwq, -clip, clip)
        dwk = np.clip(dwk, -clip, clip)
        dwv = np.clip(dwv, -clip, clip)
        dq  = np.clip(dq,  -clip, clip)
        dk  = np.clip(dk,  -clip, clip)
        dv  = np.clip(dv,  -clip, clip)
        
        self.qw -= lr * dwq
        self.kw -= lr * dwk
        self.vw -= lr * dwv

        dx = dq @ self.qw.T + dk @ self.kw.T + dv @ self.vw.T
        

        return dx
        
            
            
            
                    