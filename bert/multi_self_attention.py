import numpy as np
from mln import mask

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


class msa:
    def __init__(self,input_size,num_heads,vocab):
        self.w=None
        self.q=None
        self.v=None
        dk=input_size/num_heads
        self.dk=dk
        self.input_size=input_size
        self.kw=np.random.rand(input_size,dk)*0.01
        self.vw=np.random.rand(input_size,dk)*0.01
        self.qw=np.random.rand(input_size,dk)*0.01
    def forward(self,x,mask_vec,vocab):
        x=np.array(x)
        
        masked,target=mask(x,vocab,mask_vec)
        
        
        dk=self.kw.shape[1]
        out=[]
        weight=[]
        arrayv=[]
        
        for i in range(len(x)):
            tok=masked[i]
            q=tok @ self.qw
            z=np.zeros(dk)
            scores = np.array([q @ (masked[j] @ self.kw) / np.sqrt(dk) for j in range(len(x))])
            weights = softmax(scores)
            vec=[]
            for j in range(len(x)):
                
                
                v=masked[j]@self.vw
                vec.append(v)
                z=z+weights[j]*v
            out.append(z)
            arrayv.append(vec)
            weight.append(weights)
            
        return np.array(out),target,weight,arrayv
    def backdrop(self,x,weights,z,arrayv):
        dwv=np.zeros((self.input_size,self.dk))
        dwq=np.zeros((self.input_size,self.dk))
        dwk=np.zeros((self.input_size,self.dk))
        
        dq=np.zeros_like(self.dk)
        dk=np.zeros_like(self.dk)
        da=np.zeros_like(len(weights))
        
        
        
        for i in range(len(weights)):
            dv+=weights[i]*z
            dwv=np.outer(x[i],dv)
            da[i]=z@arrayv[i]
        ds = weights*(da - np.sum(da * weights))
        for i in range(len(ds)):
            dq+=ds[i]    
            
            


                                    
                
            
            
        
            