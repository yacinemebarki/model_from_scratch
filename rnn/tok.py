import numpy as np

def normalization(text):
    return text.lower().split()

class tokenizer:
    def __init__(self):
        wordid={}
        wordid["[PAD]"]=0
        wordid["[MASK]"]=1
        self.wordid=wordid
        self.vecword={}
        self.idx=2
    def fit(self,texts):
        for text in texts:
            text=normalization(text)
            for word in text:
                if word not in self.wordid:
                    self.wordid[word]=self.idx
                    
                    
                    self.idx+=1
                    
    def encode(self,text):
        
        result=[]
        for te in text:
            te=normalization(te)
            a=[]
            for word in te:
                if word in self.wordid:
                    a.append(self.wordid[word])
                  
            result.append(a)    
            
        return result
    def padding(self,text_vec,maxlen):
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
    
class embedding:
    def __init__(self,wordid,out_dim):
        self.vecword={}
        self.wordid=wordid
        self.type="embedding" 
        self.out_dim=out_dim   
    def embedding_tran(self):
        for word in self.wordid:
            self.vecword[self.wordid[word]]=np.random.rand(self.out_dim)*0.1
    def forward(self,vec_text):
        
        vec_text=np.array(vec_text) 
        
        

    
        if vec_text.ndim > 1:
            vec_text = vec_text.flatten()
       
        
        vec=[]
        for word in vec_text:
            if word==0:
                vec.append(np.zeros(len(next(iter(self.vecword.values())))))
            else :
                if word in self.vecword:
                    vec.append(self.vecword[word])
                else :
                    vec.append(np.random.rand(self.out_dim)*0.1)    
            
        return np.array(vec)            
                    
                
 
 