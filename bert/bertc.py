import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoder import enco
from rnn.tok import tokenizer,embedding
from mln import mask

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 


class bert:
    def __init__(self,n_encoder,input_size,num_heads,n_token,emb):
        self.input_size=input_size
        self.n_encoder=n_encoder
        self.layers=[]
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
        for epoch in range(100):
            
            for i in range(n_samples):
                tokens=x[i]
                masked_tokens, target=mask(tokens,self.wordid)
                a = emb.forward(masked_tokens)
                for l in self.layers:
                    
                    a=l.forward(a)
            
                out=a@self.w_vocab+self.b_vocab
                out=softmax(out)
            
                z=np.zeros_like(out)
            
                for j,id in enumerate(target):
                    print("the id ",id)
                    if (id!=-1):
                        one_hot= np.zeros(self.n_token)
                        one_hot[id]=1
                        z[j]=out[j]-one_hot
                print("the lost",z)        
            
                dw=a.T @ z
                db=z.sum(axis=0)
                dout=z@self.w_vocab.T
                self.w_vocab-=lr*dw
                self.b_vocab-=lr*db
                print("dout",dout)
            
                for l in reversed(self.layers):
                    dout=l.backdrop(dout,lr)
        
    def predict(self,x):
        x=np.array(x)
        n_samples=x.shape[0]
        out=[]
        for i in range(n_samples):
            a=x[i]
            a = emb.forward(a)
            for l in self.layers:
                a=l.forward(a)
            prob=a@self.w_vocab+self.b_vocab
            print("logits variance:", np.var(prob))
            prob=softmax(prob)
            
            
            pred_tokens=np.argmax(prob, axis=1)
            out.append(pred_tokens)
        return out    
                
                
            



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
train_data = text_array * 100  
vec = tok.encode(train_data)
vec = tok.padding(vec, 7)
train_data = text_array * 100  
vec = tok.encode(train_data)
vec = tok.padding(vec, 7)
vec=tok.encode(text_array)
vec=tok.padding(vec,7)
emb=embedding(tok.wordid,7)
emb.embedding_tran()

print(vec[0])


ber=bert(2,7,3,len(tok.wordid),emb)


ber.fit(vec,0.01)   

predict_text = [
    "I love [MASK]",                      # predict what comes after "I love"
    "Deep learning is [MASK]",            # predict missing word
    "Python is [MASK]",                    # predict "great" or similar
    "I enjoy coding in [MASK]",           # predict "Python"
    "Neural networks are [MASK]",         # predict "interesting" or similar
    "I hate [MASK] in my code",           # predict "bugs"
    "Machine learning is [MASK]",         # predict "amazing"
    "Training models is [MASK]",          # predict "rewarding"
    "Sometimes programming is [MASK]",    # predict "stressful"
    "I love [MASK] problems"              # predict "solving"
]
vec_pre=tok.encode(predict_text)
vec_pre=tok.padding(vec_pre, 7)

pred=ber.predict(vec_pre)
print(pred)








