import numpy as np

class recurent:
    def __init__(self,H_size):
        self.hidden=[]
        self.bh=np.zeros(H_size)
        
        self.H_size=H_size
        self.wh=np.random.rand(H_size,H_size)* np.sqrt(1 / H_size)
        self.h=np.zeros(H_size)*0.01 
        self.w=None
        self.hidden.append(self.h)
        self.type="recurente"
        
    def forward(self,x):
        if len(self.hidden)==1:
            self.w=np.random.rand(x.shape[0],self.H_size)*np.sqrt(1 / x.shape[0])
           
        h_t=np.tanh(x @ self.w +self.h @ self.wh +self.bh)
        self.h=h_t
        self.hidden.append(h_t)
        return np.array(h_t)
    def backdrop(self,dout,lr,x,dh_next,t):
        
        
        
        dh_t=dout+dh_next
        dtan=dh_t *(1-self.hidden[t]**2)
        dw_x=x[:,None] @ dtan[None,:]
        dw_h = np.zeros_like(self.wh)
        if t>0:
            dw_h=self.hidden[t-1][:,None] @ dtan[None,:]
        dbh=dtan
        max_norm = self.H_size


        dw_x = np.clip(dw_x, -max_norm, max_norm)
        dw_h = np.clip(dw_h, -max_norm, max_norm)
        dbh = np.clip(dbh, -max_norm, max_norm)

        self.wh-=lr*dw_h
        self.bh-=lr*dbh
        self.w-=lr*dw_x

           
             
        dh_next = dtan @ self.wh.T
        return dh_next
   
            
        
        
                