import numpy as np 
import pandas as pd 
class node:
    def __init__(self,feature=None,lable=None,left=None,right=None):
        self.feature=feature
        self.left=left
        self.right=right
        self.lable=lable
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
 
