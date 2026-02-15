from .flatt import flatt
from .conv import conv
from .maxpool import maxpool
import numpy as np
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)
class layerc :
    def __init__(self):
        self.layers=[]
        self.wout=None
        self.bout=None
    def addconv(self,n_kernel,kernel_size,input_shape,stride):
        convlayer=conv(n_kernel,kernel_size,input_shape,stride)
        self.layers.append(convlayer)
    def addmaxpool(self,pool_size,stirde):
        maxpoollayer=maxpool(pool_size,stirde)
        self.layers.append(maxpoollayer)
    def addflatt(self):
        flattlayer=flatt()
        self.layers.append(flattlayer)
    def fit(self,x,y,learning_rate=0.01,epoches=1000):
        
        x=np.array(x,dtype=np.float64)
        y=np.array(y)
        n_samples=x.shape[0]
        n_class=len(np.unique(y))
        y_onehot=np.eye(n_class)[y]
        if x.ndim == 3:
            x = x[..., np.newaxis]

        
        
        
        for epoch in range(epoches):
            for t in range(n_samples):
                a=x[t]
                print("input",a.shape)
                for l in self.layers:
                    if l.type=="maxpool":
                        print("convoutput",a)
                    a=l.forward(a)
                if t==0 and epoch==0:
                    print(a)
                    self.wout=np.random.rand(a.size,n_class)*0.1 
                    self.bout=np.zeros(n_class)
                    print("wout",self.wout)
                zout= a @ self.wout +self.bout
                aout=softmax(zout) 
                dout=aout-y_onehot[t]
                
                dwout=np.outer(a,dout)

                print("dout",dout)
                daprev=dout @ self.wout.T 
                self.wout-=learning_rate*dwout
                self.bout-=learning_rate*dout
                dout=daprev

                 
                for l in reversed(self.layers):
                    
                    print(l.type)
                    print(dout.shape)
                    print(dout)
                    dout=l.backdrop(dout,learning_rate)
    def predict(self,x):
        x=np.array(x)
        n_samples=x.shape[0]
        output=[]
        for i in range(n_samples):
            out=x[i]
            if out.ndim == 1:
                size = int(np.sqrt(out.size))
                out = out.reshape(size, size, 1)
            if out.ndim==2:
                out = out[..., np.newaxis]
            for l in self.layers:
                out=l.forward(out)
            out_flat = out.flatten()
            z = out_flat @ self.wout + self.bout
            
            aout=softmax(z)    
            output.append(aout) 
        return output                             



    