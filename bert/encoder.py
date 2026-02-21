from fnn import fnn
from multi_self_attention import msa
import numpy as np

class enco:
    def __init__(self,input_size,num_heads):
        self.flayer=fnn(input_size)
        self.mlayer=msa(input_size,num_heads)

    def forward(self,x,mask_vec,vocab,lr):
        out,target=self.mlayer.forward(x,mask_vec,vocab)
        
        result=self.flayer.forward(out)
        return result,target
    def backdrop(self,dout,target,lr,i) :
        for i in range(len(dout)):
            d1=self.flayer.backdrop(dout,lr,i)
            d2=self.mlayer.backdrop(d1,lr,i)
        
           
  
            
            
        
        
        
        
        
        
            
        