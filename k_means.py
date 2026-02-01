import numpy as np
def k_means(x,k,max_iters=10):
    n_samples, n_features = x.shape
    idx=np.random.choice(n_samples,k,replace=False)
    centroids=x[idx]
    for i in range(max_iters):
        labels=np.zeros(n_samples)
        for j in range(n_samples):
            
            distances=np.linalg.norm(x[j]-centroids[0])
            cluster=0
            for c in range(1,k):
                dist=np.linalg.norm(x[j]-centroids[c])
                if dist<distances:
                    distances=dist
                    cluster=c
            labels[j]=cluster
        for c in range(k):
            points=x[labels==c]
            if len(points)>0:
                centroids[c]=np.mean(points,axis=0)
    return labels,centroids                
                        



