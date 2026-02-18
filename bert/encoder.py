from fnn import fnn
from multi_self_attention import msa
import numpy as np

class enco:
    def __init__(self,input_size,num_heads):
        self.flayer=fnn(input_size)
        self.mlayer=msa(input_size,num_heads)
    def forward(self,x,mask_vec,vocab):
        out,target=self.mlayer.forward(x,mask_vec,vocab)
        
        result,vec,grad_vec=self.flayer.forward(out)
        
        for i in target:
            z=target[i]-result[i]
            loss=-np.log(result[i])
            
        
        
        
        
        
        
            
        