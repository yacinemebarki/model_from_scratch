from fnn import fnn
from multi_self_attention import msa
import numpy as np

class enco:
    def __init__(self,input_size,num_heads):
        self.flayer=fnn(input_size)
        self.mlayer=msa(input_size,num_heads)

    def forward(self,x):
        out=self.mlayer.forward(x)
        out=self.flayer.forward(out)
        return out
    def backdrop(self,dout,lr):
        dx=self.flayer.backdrop(dout,lr)
        dx=self.mlayer.backdrop(dx,lr)
        return dx
            
           
  
            
            
        
        
        
        
        
        
            
        