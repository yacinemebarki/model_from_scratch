import numpy as np

def mask(x,vocab):
    x=np.array(x)

    n_token=len(x)
    maskidx=np.random.choice(n_token,size=int(0.15*n_token),replace=False)
    masked=x.copy()
    print(x)
    target=np.full(len(x), -1)
    
    
    
    for i in maskidx:
        prob=np.random.rand()
        
        target[i]=x[i]
        if prob<0.8:
            
            masked[i]=-1
        elif prob<0.9:
            pass
        else:
            masked[i]=np.random.choice(list(vocab.values()))   
    return masked,target
        