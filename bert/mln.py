import numpy as np

def mask(x,vocab):
    x=np.array(x)
    validx=np.where(x != 0)[0]
    if len(validx) == 0:
        return x.copy(), np.full(len(x), -1)

    n_token=len(x)
    maskidx=np.random.choice(validx,size=max(1,int(0.3*len(validx))),replace=False)
    masked=x.copy()
    
    target=np.full(len(x), -1)
    
    
    
    for i in maskidx:
        prob=np.random.rand()
        
        target[i]=x[i]
        
        if prob<0.8:
            
            masked[i]=vocab["[MASK]"]
        elif prob<0.9:
            pass
        else:
            masked[i]=np.random.choice(list(vocab.values()))   
    return masked,target
        