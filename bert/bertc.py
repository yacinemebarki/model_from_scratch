import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoder import enco
from rnn.tok import tokenizer,embedding

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z) 


class bert:
    def __init__(self,n_encoder,input_size,num_heads,n_token,mask_vec,vocab,wordid):
        self.input_size=input_size
        self.n_encoder=n_encoder
        self.layers=[]
        self.n_token=n_token
        self.wordid=wordid
        self.w_vocab=np.random.randn(input_size, n_token) * 0.01
        self.b_vocab=np.zeros(n_token)
        self.mask_vec=mask_vec
        self.vocab=vocab
    
        for i in range(n_encoder):
            l=enco(input_size,num_heads)
            self.layers.append(l)
    
    
    def fit(self,x,lr):
        x=np.array(x)
        n_samples=len(x)
        for i in range(n_samples):
            a=x[i]
            for l in self.layers:
                print("enter",i)
                a,target=l.forward(a,mask_vec,self.vocab,self.wordid)
            
            out=a@self.w_vocab+self.b_vocab
            out=softmax(out)
            
            z=np.zeros_like(out)
            
            for j,id in enumerate(target):
                
                if (id!=0).any():
                    one_hot=np.zeros(self.n_token)
                    one_hot[id]=1
                    z[j]=out[j]-one_hot
            
            dw=a.T @ z
            db=z.sum(axis=0)
            dout=z@self.w_vocab.T
            self.w_vocab-=lr*dw
            self.b_vocab-=lr*db
            
            for l in reversed(self.layers):
                dout=l.backdrop(dout,lr)
        return self.w_vocab,self.b_vocab



#test

text_array = [
    "I love AI",
    "Deep learning is fun",
    "Hello world",
    "Python is great",
    "RNN is powerful",
    "I love deep learning",
    "Machine learning is amazing",
    "I enjoy coding in Python",
    "Artificial intelligence is the future",
    "Neural networks are interesting",
    "I hate bugs in my code",
    "Debugging is frustrating",
    "Syntax errors are annoying",
    "Sometimes programming is stressful",
    "I dislike slow computers",
    "I love solving problems",
    "Data science is fascinating",
    "I enjoy learning new algorithms",
    "Training models is rewarding",
    "I hate runtime errors",
    "Optimization is challenging",
    "I like experimenting with models",
    "Python makes programming easier",
    "I am learning deep learning",
    "I dislike complicated setups",
    "I enjoy clean code",
    "Machine learning can be tricky",
    "I love AI research",
    "Sometimes training takes too long",
    "I like visualizing data",
    "RNNs can remember sequences",
    "I hate missing semicolons",
    "I enjoy writing functions",
    "I dislike long debugging sessions"
]
tok=tokenizer()
tok.fit(text_array)
vec=tok.encode(text_array)
vec=tok.padding(vec,7)
emb=embedding(tok.wordid,7)
emb.embedding_tran()
emb_vec=[]

for text in vec:
    a=emb.forward(text)
    emb_vec.append(a)
print(emb_vec)

print(len(emb_vec[0]))
mask_vec=np.random.rand(7)*0.1

ber=bert(2,7,3,len(tok.wordid),mask_vec,emb.vecword,tok.wordid)


w,b=ber.fit(emb_vec,0.01)   
print(w,b)