import numpy as np
import re
import pandas as pd
def decompose(text):
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]', '', text)
    text=re.sub(r'\s+', ' ', text)
    return text.strip().split()
print(decompose("Hello, World! hello This is a test."))
def compute_tf(data):
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
    
    for freq in tf:
        a=[]
    
        for word in doc_words:
            val=float(freq[word] * idf[word]) if word in freq else 0.0
            a.append(val)
            
        tfidf.append(a)
    return tfidf
print(compute_tf(["this is a sample","this is another example example example"]))
text=["are you learning machine learning",
      "machine learning is fun",
      "I love coding in python",
      "python is a great programming language",
      "do you love deep learning",
      "i hate bugs in my code",
      "i hate information systems"]
tfidf_result=compute_tf(text)
print(tfidf_result)         
      


    
            
