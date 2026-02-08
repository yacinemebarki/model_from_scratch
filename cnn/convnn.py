from flatt import flatt
from conv import conv
from maxpool import maxpool
import numpy as np
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)
class layer :
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
        x=np.array(x)
        y=np.array(y)
        n_samples=x.shape[0]
        n_class=len(np.unique(y))
        y_onehot=np.eye(n_class)[y]
        x=x[..., np.newaxis]
        for epoch in range(epoches):
            for t in range(n_samples):
                a=x[t]
                for l in self.layers:
                    a=l.forward(a)
                if t==0 and epoch==0:
                    self.wout=np.random.rand(a.size,n_class)*0.01 
                    self.bout=np.zeros(n_class)
                zout= a @ self.wout +self.bout
                aout=softmax(zout) 
                dout=aout-y_onehot[t]
                
                dwout=np.outer(a,dout)
                daprev=dout @ self.wout.T 
                self.wout-=learning_rate*dwout
                self.bout-=learning_rate*dout
                 
                for l in reversed(self.layers):
                    print(l.type)
                    dout=l.backdrop(dout,learning_rate)
#test cnn
x=np.array([
    [[1, 2, 1, 0],
     [0, 1, 0, 2],
     [2, 1, 0, 1],
     [1, 0, 2, 1]],

    [[2, 0, 1, 1],
     [1, 1, 0, 2],
     [0, 2, 1, 0],
     [1, 1, 2, 1]]
])

y=np.array([0, 1])  



model=layer()


model.addconv(n_kernel=1, kernel_size=(3,3), input_shape=(4,4), stride=1)
model.addmaxpool(pool_size=(2,2), stirde=2)
model.addflatt()

model.fit(x, y, learning_rate=0.01, epoches=5)

print("Training done!")
print("Output weights:", model.wout)
print("Output bias:", model.bout)

    