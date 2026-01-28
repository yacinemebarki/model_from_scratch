import numpy as np 
import pandas as pd 
class node:
    def __init__(self,feature=None,lable=None,left=None,right=None,threshold=None,value=None):

        self.feature=feature
        self.left=left
        self.right=right
        self.lable=lable
        self.threshold=threshold
        self.value=value
def entropy(p):
    if len(p)==0:
        raise ValueError("empty array")
    s=0
    s2=0
    for val in p:
        if val==0:
            s=s+1
        else :
            s2=s2+1
    q0=s/len(p)
    q1=s2/len(p)
    if q0==0 or q1==0:
        return 0
    ent=-q0*np.log2(q0)-q1*np.log2(q1)
    return ent
def information_gain(parent_fea,y_lab):
    if len(parent_fea)==0 or len(y_lab)==0:
        raise ValueError("empty array")
    left_x = []
    right_x = []
    s0=0
    s1=0
    left_x = []
    right_x = []
    left_y = []
    right_y = []
    for i in range(len(parent_fea)):
        if parent_fea[i] == 0:
            left_x.append(parent_fea[i])
            left_y.append(y_lab[i])
        else:
            right_x.append(parent_fea[i])
            right_y.append(y_lab[i])

    
    p0 = len(left_x) / len(y_lab)
    p1 = len(right_x) / len(y_lab)
    if len(right_y) == 0 or len(left_y) == 0:
        return 0, left_x, left_y, right_x, right_y
        

    ig = entropy(y_lab) - p0 * entropy(left_y) - p1 * entropy(right_y)
    return ig, left_x, left_y, right_x, right_y
                        
def fit(x,y):
    if len(x)==0 or len(y)==0:
        raise ValueError("empty array")
    if len(x)!=len(y):
        raise ValueError("different lengths")
    root=node()
    root.feature=x
    root.lable=y
    def split(root):
        
        if len(root.feature)==0:
            return None
        

        
        
        if len(np.unique(root.lable)) == 1:
            leaf = node()
            leaf.lable = [root.lable[0]]  
            leaf.left = None
            leaf.right = None
            return leaf
        ig,t0,y0,t1,y1=information_gain(root.feature,root.lable)
        if ig == 0 or len(y0) == 0 or len(y1) == 0:
            leaf = node()
            leaf.lable = [np.bincount(root.lable).argmax()]  
            leaf.left = None
            leaf.right = None
            return leaf
        print("X:", t0)
        print("y:", y0)
        print("X:", t1)
        print("y:", y1)
        print("entropy:", entropy(y))
        
        print("IG:", ig)
        print("t0:", t0)
        print("t1:", t1)
        
        node_left=node()
        node_right=node()
        node_left.feature=t0
        node_left.lable=y0
        node_right.feature=t1
        node_right.lable=y1
        root.left=split(node_left)
        root.right=split(node_right)
        return root
        
    return split(root)
def print_tree(root):
    if root is None:
        return
    if len(np.unique(root.lable)) == 1:
        print("Leaf Node: ", root.lable)
    else:
        print("Node")
        print_tree(root.left)
        print_tree(root.right)
def MSE(y_left,y_right):
    n=len(y_left)+len(y_right)
    if n==0:
        raise ValueError("empty array")
    mean_left=np.mean(y_left) if len(y_left)>0 else 0
    mean_right=np.mean(y_right) if len(y_right)>0 else 0
    mse_left=sum((y_left - mean_left) ** 2) if len(y_left)>0 else 0
    mse_right=sum((y_right - mean_right) ** 2) if len(y_right)>0 else 0
    mse=(mse_left + mse_right)/n
    return mse  
def fit_regression(X, y, max_depth=float('inf'), min_samples=2):
    X = np.array(X)
    y = np.array(y)

    n_samples,n_features=X.shape

    
    

    best_mse = float('inf')
    best_feature = None
    best_threshold = None
    best_split = None

    for feature in range(n_features):
        x_feature=X[:, feature]
        sorted_idx=np.argsort(x_feature)
        x_sorted=x_feature[sorted_idx]
        y_sorted=y[sorted_idx]
        for i in range(n_samples - 1):
            threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
            left_mask = x_feature <= threshold
            right_mask = x_feature > threshold
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            mse = MSE(y[left_mask], y[right_mask])
            if mse<best_mse:
                best_mse=mse
                best_feature=feature
                best_threshold=threshold
                best_split=(left_mask, right_mask)

    if max_depth==0 or n_samples<=min_samples or best_mse==0:
        leaf = node()
        leaf.value = np.mean(y)
        return leaf
    if best_feature is None:
        leaf = node()
        leaf.value = np.mean(y)
        return leaf

    root=node()
    root.feature=best_feature
    root.threshold=best_threshold

    left_mask, right_mask=best_split

    root.left = fit_regression(X[left_mask], y[left_mask], max_depth - 1, min_samples)
    root.right = fit_regression(X[right_mask], y[right_mask], max_depth - 1, min_samples)

    return root


def print_tree_regression(root):
    if root is None:
        return
    if root.left is None and root.right is None:
        print("Leaf Node value: ", root.value)
    else:
        print("Node with threshold: ", root.threshold)
        print_tree_regression(root.left)
        print_tree_regression(root.right)





 
