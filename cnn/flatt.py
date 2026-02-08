import numpy as np

class flatt:
    def __init__(self):
        
        
        self.type="flatt"

        
    def forward(self,input):
        self.original_shape=input.shape
        

        return input.reshape(-1)
    def backdrop(self,da,lr):
        return da.reshape(self.original_shape)
    

