import numpy as np
import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from decoder import deco
from rnn.tok import tokenizer,embedding

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) 

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
            a=self.emb.forward(a)
            for l in self.layer:
                a=l.forward(a)
            
            
            out=a@self.w_out+self.b_out
            out=softmax(out)
            
            
            z=out-target
            dw=a.T@z
            db=z.sum(axis=0)
            dout=z@self.w_out.T
            self.w_out-=lr*dw
            self.b_out-=lr*db
            
            
            for l in reversed(self.layer):
                dout=l.backdrop(dout,lr)
                
                
        return self.w_out        
                
                
            
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
    "I dislike long debugging sessions",
    "I love playing football with my friends",
    "The weather today is sunny and warm",
    "She enjoys reading books in the library",
    "We are going to watch a movie tonight",
    "He likes to eat pizza for lunch",
    "Python is a popular programming language",
    "They visited the museum last weekend",
    "I am learning machine learning and AI",
    "My dog loves to run in the park",
    "The cat is sleeping on the sofa",
    "We are planning a trip to the mountains",
    "She bought a new pair of shoes yesterday",
    "The sun rises in the east and sets in the west",
    "He is studying computer science at university",
    "I enjoy listening to music while studying",
    "They are playing basketball in the gym",
    "The children are drawing pictures in class",
    "My favorite color is blue",
    "The train leaves at 9 o'clock every morning",
    "I need to finish my homework before dinner",
    "She is cooking pasta for dinner tonight",
    "We went hiking in the forest last weekend",
    "He is writing a book about history",
    "The flowers in the garden are blooming",
    "I like drinking coffee in the morning",
    "They are watching a football match on TV",
    "My brother is learning to play the guitar",
    "The movie was very exciting and fun",
    "I visited my grandparents last summer",
    "She is practicing yoga every day",
    "We are celebrating my friend's birthday",
    "He likes painting landscapes in his free time",
    "The car is parked in front of the house",
    "I am studying mathematics and physics",
    "The dog is barking at the mailman",
    "They are building a treehouse in the backyard",
    "She enjoys swimming in the ocean",
    "We are planning to go to the beach tomorrow",
    "He bought a new laptop for work",
    "I like to eat ice cream in the summer",
    "The children are playing in the playground",
    "She is reading a novel by her favorite author",
    "We went to a concert last night",
    "He is learning French and Spanish",
    "I enjoy hiking in the mountains",
    "They are watching a documentary on animals",
    "My cat likes to chase birds in the garden",
    "She is learning how to bake cakes",
    "We are organizing a charity event next month",
    "He is fixing the bike in the garage",
    "I love painting and drawing in my free time",
    "The sun is shining brightly in the sky",
]

tok=tokenizer()

tok.fit(text_array)
train_data = text_array * 100  
vec = tok.encode(train_data)
vec = tok.padding(vec, 12)
emb=embedding(tok.wordid,32)
emb.embedding_tran()
gp=gpt(2,32,12,len(tok.wordid),emb)
w_out,b_out=gp.fit(vec,0.01)  
print(w_out,b_out)                 
            
        
        