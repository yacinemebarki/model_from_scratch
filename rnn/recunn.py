import numpy as np

class recurent:
    def __init__(self,H_size):
        self.hidden=[]
        self.bh=np.zeros(H_size)
        self.b=np.zeros(H_size)
        self.H_size=H_size
        self.wh=np.random.rand(H_size,H_size)*0.01
        self.h=np.zeros(H_size)*0.01
        self.w=None
        self.hidden.append(self.h)
        
    def forward(self,x):
        if len(self.hidden)==1:
            self.w=np.random.rand(x.shape[0],self.H_size)*0.01
        h_t=np.tanh(x @ self.w +self.h @ self.wh +self.bh)
        self.h=h_t
        self.hidden.append(h_t)
        return h_t
    def backdrop(self,dout,lr):
        dh_next=0
        
        dh_t=dout+dh_next
            
        dtan=dh_t * (1-self.h**2)
            
        
        
                