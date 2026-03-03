import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .encoder import enco

from .mln import mask

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 


class bert:
    def __init__(self,n_encoder,input_size,num_heads,n_token,emb):
        self.input_size=input_size
        self.n_encoder=n_encoder
        self.layers=[]
        self.emb=emb
        self.n_token=n_token
        wordid={}
        
        self.wordid=emb.wordid
        self.w_vocab=np.random.randn(input_size,n_token) * 0.01
        self.b_vocab=np.zeros(n_token)
        
        self.vocab=emb.vecword
    
        for i in range(n_encoder):
            l=enco(input_size,num_heads)
            self.layers.append(l)
    
    
    def fit(self,x,lr):
        x=np.array(x)
        n_samples=len(x)
        
            
        for i in range(n_samples):
            tokens=x[i]
            masked_tokens, target=mask(tokens,self.wordid)
                
                
            a = self.emb.forward(masked_tokens)
                
                
            for l in self.layers:
                    
                a=l.forward(a)
            
            out=a@self.w_vocab+self.b_vocab
            out=softmax(out)
            
            z=np.zeros_like(out)
           
            
            for j,id in enumerate(target):
                    
                if (id!=-1):
                    one_hot= np.zeros(self.n_token)
                    one_hot[int(id)]=1
                    z[j]=out[j]-one_hot
                        
                       
            
            dw=a.T @ z
            db=z.sum(axis=0)
            dout=z@self.w_vocab.T
            self.w_vocab-=lr*dw
            self.b_vocab-=lr*db
                
            
            for l in reversed(self.layers):
                dout=l.backdrop(dout,lr)
            self.emb.backward(dout,masked_tokens,lr)    
            
            
        
    def predict(self,x):
        x=np.array(x)
        n_samples=x.shape[0]
        out=[]
        for i in range(n_samples):
            a=x[i]
            a = self.emb.forward(a)
            
            for l in self.layers:
                a=l.forward(a)
               
            
            mask_idx = np.where(np.array(x[i])==1)[0][0]
            print("the mask id",mask_idx)  
            masked_vec = a[mask_idx]
             
            
            prob=masked_vec@self.w_vocab+self.b_vocab
            
            
            prob = np.exp(prob - np.max(prob))
            prob = prob / np.sum(prob)
            print("probs",prob)

            pred_token = np.argmax(prob)
            
            out.append(pred_token)
        return out    
                
                
            



#test


    









