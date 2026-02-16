import numpy as np

def relu(x):
    return np.maximum(0,np.array(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class fnn:
    def __init__(self,input_size):
        self.input_size=input_size
        
        self.w=np.random.rand(self.input_size, 4*self.input_size) * 0.01
        self.b=np.zeros(4*self.input_size)

        self.w_out=np.random.rand(4*self.input_size, self.input_size) * 0.01
        self.b_out=np.zeros(self.input_size)
        

    def forward(self,x,vocab,learning_rate=0.01):
        x=np.array(x)    
        n_sample=x.shape[0]
        output=[]
        
        for t in range(n_sample):
            
            y=self.w @ x[t]+self.b
            y=relu(y)
            y2=self.w_out*y+self.b_out
            output.append(y2)
        return np.array(output)    
        
                
                
                    
            
            
                
        