def normalization(text):
    return text.lower().split()

class tokenizer:
    def __init__(self):
        self.wordid={}
        self.vecword=[]
        self.idx=1
    def fit(self,texts):
        for text in texts:
            text=normalization(text)
            for word in text:
                if word not in self.wordid:
                    self.wordid[word]=self.idx
                    
                    
                    self.idx+=1
                    
    def encode(self,text):
        text=normalization(text)
        result=[]
        for word in text:
            if word in self.wordid:
                result.append(self.wordid[word])
        return result        
def padding(text_vec,maxlen):
    result=[]
    for vec in text_vec:
        t=maxlen-len(vec)
        if t<0:
            raise ValueError("change the max len")
        v=[]
        for i in range(t):
            v.append(0)
        for j in range(t,maxlen):
            v.append(vec[j-t])
        result.append(v)
    return result 
text_array = [
    "I love AI",
    "Deep learning is fun",
    "Hello",
    "Python is great for NLP",
    "RNN LSTM Transformers"
]
       
tok=tokenizer()
tok.fit(text_array)
vec=[tok.encode(text) for text in text_array]
print("tokenization",vec)
vec_padd=padding(vec,10)
print("the padding is",vec_padd)            
            
                                        
                    
                        

    