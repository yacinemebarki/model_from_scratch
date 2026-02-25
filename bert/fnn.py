import numpy as np

def relu(x):
    return np.maximum(0,np.array(x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

class fnn:
    def __init__(self,input_size):
        self.input_size=input_size
        
        self.w=np.random.rand(self.input_size, 4*self.input_size) * 0.01
        self.b=np.zeros(4*self.input_size)

        self.w_out=np.random.rand(4*self.input_size, self.input_size) * 0.01
        self.b_out=np.zeros(self.input_size)
        

    def forward(self,x):
        x=np.array(x)  
        self.x=x  
        
        
        
        
            
        y= x@self.w +self.b
        
        y=relu(y)
        out=y@self.w_out+self.b_out
        
        self.y=y
        self.out=out
            
        return np.array(out)   
    def backdrop(self,z,lr):
        
        dw2=self.y.T@z
        da=z@self.w_out.T
        db=da*relu_derivative(self.y)
        dw=self.x.T@db
        self.w-=lr*dw
        self.w_out-=lr*dw2
        self.b_out-=lr* z.sum(axis=0)
        self.b-=lr* db.sum(axis=0)
        dx=db@self.w.T
        return dx
        
                
                
                    
            
            
                
        