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
        n_sample=x.shape[0]
        output=[]
        vec=[]
        self.gradvec=[]
        
        for t in range(n_sample):
            
            y=self.w @ x[t]+self.b
            self.gradvec.append(y)
            y=relu(y)
            y2=y@self.w_out+self.b_out
            output.append(y2)
            vec.append(y)
        self.output=output
        self.vec=vec    
            
        return np.array(output)   
    def backdrop(self,z,lr,i):
        dw2=np.outer(self.vec[i],z)
        da=self.w_out.T@z
        db=da*relu_derivative(self.output[i])
        dw=np.outer(self.x,db)
        self.w-=lr*dw
        self.w_out-=lr*dw2
        self.b_out-=lr*z
        self.b-=lr*db
        
                
                
                    
            
            
                
        