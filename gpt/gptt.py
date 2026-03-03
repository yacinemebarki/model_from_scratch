import numpy as np
import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from .decoder import deco


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 

def softmax_pred(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)
    

def to_one_hot(indices, n_classes):
    one_hot=np.zeros((len(indices), n_classes))
    one_hot[np.arange(len(indices)), indices]=1
    return one_hot

def create(x):
    inp=x[:-1]
    target=x[1:]
    return inp,target

class gpt:
    
    def __init__(self,n_decoder,input_size,num_heads,n_token,emb):
        self.input_size=input_size
        self.dk=input_size//num_heads
        self.n_token=n_token
        self.emb=emb
        self.layer=[]
        self.w_out=np.random.rand(input_size,n_token)*0.01
        self.b_out=np.zeros(n_token)
        
        for i in range(n_decoder):
            l=deco(input_size,self.dk,n_token)
            self.layer.append(l)
    
    
    def fit(self,x,lr):
        x=np.array(x)
        n_sample=x.shape[0]
        
        for i in range(n_sample):
            a=x[i]
            inp,target=create(a)
            a=self.emb.forward(inp)
            
            for l in self.layer:
                a=l.forward(a)
            
            
            out=a@self.w_out+self.b_out
            out=softmax(out)
            
            
            
            target=to_one_hot(target,self.n_token)
            
            z=out-target
            dw=a.T@z
            db=z.sum(axis=0)
            dout=z@self.w_out.T
            self.w_out-=lr*dw
            self.b_out-=lr*db
            
            
            for l in reversed(self.layer):
                dout=l.backdrop(dout,lr)
                
                
        
    def predict(self,x):
    
        
        
        
        output=[]
        
        for a in x:
            
            a=self.emb.forward(a)
            for l in self.layer:
                a=l.forward(a)
            out=a@self.w_out+self.b_out
            out=out[-1]
            out= softmax_pred(out)
            out[0]=0
            out=np.argmax(out)
            
            print(out)
            output.append(out)
        return output
                      
                
                
            
           
            
        
        