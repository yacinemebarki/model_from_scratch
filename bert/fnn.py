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
        

    def forward(self,x,vocab,learning_rate=0.01):
        x=np.array(x)    
        n_sample=x.shape[0]
        output=[]
        vec=[]
        gradvec=[]
        
        for t in range(n_sample):
            
            y=self.w @ x[t]+self.b
            gradvec.append(y)
            y=relu(y)
            y2=y@self.w_out+self.b_out
            output.append(softmax(y2))
            vec.append(y)
            
        return np.array(output),np.array(vec),np.array(gradvec)   
    def backdrop(self,x,z,a,y,lr):
        dw2=np.outer(a,z)
        da=self.w_out.T@z
        db=da*relu_derivative(y)
        dw=np.outer(x,db)
        self.w-=lr*dw
        self.w_out-=lr*dw2
        self.b_out-=lr*z
        self.b-=lr*db
        
                
                
                    
            
            
                
        