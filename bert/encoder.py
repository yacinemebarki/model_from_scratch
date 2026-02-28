from fnn import fnn
from multi_self_attention import msa
import numpy as np

class enco:
    def __init__(self,input_size,num_heads):
        self.flayer=fnn(input_size)
        self.mlayer=msa(input_size,num_heads)
        self.norm1 = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
        self.norm2 = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

    def forward(self,x):
        out1=self.mlayer.forward(x)
      
        out1 = self.norm1(out1+x) 
        out=self.flayer.forward(out1)
        
        out = self.norm2(out+out1)
        return out
    def backdrop(self,dout,lr):
        dx=self.flayer.backdrop(dout,lr)
        dx = dx + dout
        dx=self.mlayer.backdrop(dx,lr)
        dx = dx + dout
        return dx
    
            
           
  
            
            
        
        
        
        
        
        
            
        