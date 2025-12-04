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
            
        
    tfidf_result = []   
    for sent in data:
        words=decompose(sent)
        tfidf_count={}
        tf_count={}
        for word in words:
            tf_count[word]=tf_count.get(word,0)+1
        for word in tf_count:
            tfidf_count[word]= tf_count[word]*np.log10(n/(df[word]))
        tfidf_result.append(tfidf_count)
    return tfidf_result
data=["This is a sample sentence.","This sentence is another example.","TF-IDF is a useful technique."]
tfidf_values=compute_tf(data)
for i, sent_tfidf in enumerate(tfidf_values):
    print(f"Sentence {i+1} TF-IDF:", sent_tfidf)            
                


        

                



        
    

