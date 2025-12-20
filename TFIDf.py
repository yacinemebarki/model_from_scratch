import numpy as np
import re
import pandas as pd
def decompose(text):
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]', '', text)
    text=re.sub(r'\s+', ' ', text)
    return text.strip().split()
print(decompose("Hello, World! hello This is a test."))
class TFIDF:
    def __init__(self,vocab=None,idf=None):
        self.vocab=vocab
        self.idf=idf
    def compute_tf(self,data):
        tf=[]
        doc_words=[]
        for document in data:
            words=decompose(document)
            n=len(words)
            freq={}
            for word in words:
                freq[word] = freq.get(word, 0) + 1
                if word not in doc_words:
                    doc_words.append(word)
            for word in freq:
                freq[word] /= n        
            tf.append(freq)
        idf={}
        n=len(data)
        for word in doc_words:
            s=0
            for dec in tf :
                if word in dec:
                    s=s+1
            idf[word]=np.log((n+1)/(1+s))+1
        tfidf=[]
        self.idf=idf

    
        for freq in tf:
            a=[]
            for word in doc_words:
                val=float(freq[word] * idf[word]) if word in freq else 0.
                a.append(val)
            tfidf.append(a)
        self.vocab=doc_words    
        return  np.array(tfidf,dtype=float)
    def transform(self,data):
        if self.vocab is None or self.idf is None:
            print("you need to fit the model first")
            return
        tfidf=[]
        for doc in data:
            words=decompose(doc)
            n=len(words)
            freq={}
            for word in words:
                if word in self.vocab:
                    freq[word]=freq.get(word, 0) + 1
            for word in freq:
                freq[word]=freq[word]/n
            a=[]
            for word in self.vocab:
                val=freq.get(word, 0.0) * self.idf[word]
                a.append(val)

            tfidf.append(a)
        return np.array(tfidf,dtype=float)                





    
            
