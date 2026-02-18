import numpy as np

def mask(x,vocab,mask_vec):
    mask_prob=0.15
    n_token=len(x)
    maskidx=np.random.choice(n_token,size=int(0.15*n_token),replace=False)
    masked=x.copy()
    target=[]
    
    
    for i in maskidx:
        prob=np.random.rand()
        target.append(i)
        if prob<0.8:
            masked[i]=mask_vec
        elif prob<0.9:
            pass
        else:
            masked[i]=np.random.choice(vocab)    
    return masked,target
        