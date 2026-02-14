from .tok import tokenizer,embedding
from .recunn import recurent
import numpy as np

class layer:
    def __init__(self):
        self.layers=[]
        self.w_out=None
        self.b_out=None
        self.type=None
    def addembedding(self,wordid,out_dim):
        self.layers.append(embedding(wordid,out_dim))
        


    def addrecun(self,H_size):
        self.layers.append(recurent(H_size))
        
        
    def fit(self,x,y,epoches=10,learning_rate=0.01):
        x=np.array(x)
        y=np.array(y)
        n_samples=x.shape[0]
        n_class=len(np.unique(y))
        y_onehot=np.eye(n_class)[y]
        
        for epoch in range(epoches):
            
            for t in range(n_samples):
                a=x[t]
                for l in self.layers:
                    if l.type=="embedding":
                        l.embedding_tran()
                        
                        a=l.forward([a])
                        a=a[-1]
                    else :
                        p=l    
                        a=l.forward(a) 
                       
                
                if epoch==0 and t==0:
                    self.w_out=np.random.rand(n_class,p.H_size)*0.01
                    self.b_out=np.zeros(n_class)
                z=self.w_out @ p.hidden[-1] +self.b_out
                y_pred=np.tanh(z)
                dz=y_pred-y_onehot[t]
                dw=dz[:,None] @ p.hidden[-1][None,:]
                dout=self.w_out.T @ dz

                for i in reversed(range(len(self.layers))):
                    l=self.layers[i]
                    if l.type=="recurente":
                        dh_next = np.zeros(l.H_size)
                        for h in reversed(range(len(l.hidden))):
                            dh_next=l.backdrop(dout,learning_rate,x[t],dh_next,h)
           
            
                self.w_out-=learning_rate*dw
                self.b_out-=learning_rate*dz
    def predict(self,x):
        x=np.array(x)
        n_samples=x.shape[0]
        result=[]
        for i in range(n_samples):
            a=x[i]
            for l in self.layers:
                
                if l.type=="embedding":
                    
                    a=l.forward([a])
                    a=a[-1]
                else :
                    p=l    
                    a=l.forward(a)
            z=self.w_out @ p.hidden[-1] +self.b_out
            y_pred=np.tanh(z)
            result.append(np.argmax(y_pred))
        return np.array(result)    
                     
                        
                
                             
            
            
            
            

text_array=[
    "I love AI",
    "Deep learning is fun",
    "Hello world",
    "Python is great",
    "RNN is powerful",
    "I love deep learning"
]


labels=[1, 1, 0, 1, 0, 1]

tok=tokenizer()
tok.fit(text_array)
vec=tok.encode(text_array)
vec_padded = tok.padding(vec, 5)


print("tokenization",vec)
model=layer()
model.addembedding(tok.wordid,5)
model.addrecun(6)
model.fit(vec_padded,labels)
print("wight",model.w_out)
print("bias",model.b_out)
text_pre=[
    "i love python",
    "i love machine learning"
]
vec_pre=tok.encode(text_pre)
vec_pre=tok.padding(vec_pre,5)
result=model.predict(vec_pre)
print(result)
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

# Binary labels: 1 = positive / interested, 0 = negative / frustrated
labels = [
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 0,
    0, 1, 1, 1, 0,
    1, 0, 1, 0, 1,
    1, 0, 1, 0
]

tok2=tokenizer()
tok2.fit(text_array)
vec_arr=tok2.encode(text_array)
vec_arr=tok2.padding(vec_arr,7)
print("the padding",vec_arr)


model2=layer()
model2.addembedding(tok2.wordid,7)
model2.addrecun(5)
model.addrecun(7)

model2.fit(vec_arr,labels)
print("model2 weights",model2.w_out)
print("model2 biases",model2.b_out)