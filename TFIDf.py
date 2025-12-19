import numpy as np
import re
import pandas as pd
def decompose(text):
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]', '', text)
    text=re.sub(r'\s+', ' ', text)
    return text.strip().split()
print(decompose("Hello, World! This is a test."))
def compute_tf(data):
    tf={}
    df={}
    n=len(data)
    for i in range(n):
        word=set(decompose(data[i]))
        for j in word:
            df[j]=df.get(j,0)+1
            
    vocab=list(df.keys()) 
    word_to_index={word: i for i, word in enumerate(vocab)}   
    tfidf_result = []   
    for sent in data:
        words=decompose(sent)
        tf_count={}
        i=0
        tfidf_count=np.zeros(len(vocab))
        for word in words:
            tf_count[word]=tf_count.get(word,0)+1
        for word in tf_count:
            i=word_to_index[word]
            tfidf_count[i]=tf_count[word]*np.log10(n/(df[word]))
            
        tfidf_count=np.array(tfidf_count)
        tfidf_result.append(tfidf_count)
    return tfidf_result,vocab


    
            
