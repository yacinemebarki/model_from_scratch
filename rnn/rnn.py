from tok import tokenizer
from recunn import recurent

class layer:
    def __init__(self):
        self.layers=[]
        self.w_out=None
        self.b_out=None
    def addembedding(self):
        self.layers.append(tokenizer())
    def addrecun(self,H_size):
        self.layers.append(recurent(H_size))    
            