import numpy as np

from encoder import enco

class bert:
    def __init__(self,n_encoder,input_size,num_heads,vocab):
        self.input_size=input_size
        self.n_encoder=n_encoder
        self.layers=[]
        self.w_vocab=np.random.randn(input_size, len(vocab)) * 0.01
        self.b_vocab=np.zeros(len(vocab))
    
        for i in range(n_encoder):
            l=enco(input_size,num_heads)
            self.layers.append(l)
    
    
    def forward(self,x,vocab,mask_vec,epoch=100,lr=0.01):
        x=np.array(x)
        n_samples=x.shape[0]
        
        for i in range(epoch):
            for j in range(n_samples):
                a=x[j]
                for l in self.layers:
                    a=l.forward(a,mask_vec,vocab,lr)
                logits=a@self.w_vocab + self.b_vocab 
                   
                    
                
            
            
            
        
        
    