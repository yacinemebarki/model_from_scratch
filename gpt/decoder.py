import numpy as np
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from bert.fnn import fnn
from masked_self_attention import mssa

class deco:
    def __init__(self,input_size,dk,n_tokens):
        self.flayer=fnn(input_size)
        self.mlayer=mssa(input_size,dk,n_tokens)
        self.norm1 = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
        self.norm2 = lambda x: (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
    def forward(self,x):
        out1=self.mlayer.forward(x)
        out1=self.norm1(out1+x)
        out2=self.flayer.forward(out1) 
        out2=self.norm2(out2+out1)
        return out2   
        


