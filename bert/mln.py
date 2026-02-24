import numpy as np

def mask(x,mask_vec,vocab,wordid):
    x=np.array(x)

    n_token=len(x)
    maskidx=np.random.choice(n_token,size=int(0.15*n_token),replace=False)
    masked=x.copy()
    target=np.zeros((len(x), len(x[0])),dtype=float)
    
    
    for i in maskidx:
        prob=np.random.rand()
        target[i]=x[i]
        if prob<0.8:
            print("data",x)
            print("bug1",len(masked[i]))
            masked[i]=mask_vec
        elif prob<0.9:
            pass
        else:
            masked[i]=np.random.choice(vocab)    
    return masked,target
        