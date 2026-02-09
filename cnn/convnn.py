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


model.addconv(n_kernel=1, kernel_size=(3,3), input_shape=(4,4,1), stride=1)
model.addmaxpool(pool_size=(2,2), stirde=2)
model.addflatt()

model.fit(x, y, learning_rate=0.01, epoches=5)

print("Training done!")
print("Output weights:", model.wout)
print("Output bias:", model.bout)
x2=[[[1, 3, 1, 2],
     [0, 1, 0, 0],
     [2, 1, 0, 1],
     [2, 0, 1, 0]]]
y2=model.predict(x2)
print("prediction:",y)


#testing using tensorflow dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train))
x_test=x_test[:100]
print(len(np.unique(x_test)))
x_train = x_train.reshape(-1,28, 28, 1)  
x_test  = x_test.reshape(-1,28, 28, 1) 

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0
x_train=x_train[:100]
y_train=y_train[:100]
model2=layer()
model2.addconv(n_kernel=1, kernel_size=(3,3), input_shape=(28,28,1), stride=1)
model2.addmaxpool(pool_size=(2,2), stirde=2)
model2.addflatt()
model2.fit(x_train,y_train,learning_rate=0.01,epoches=10)
print("keras weight",model2.wout)
print("keras bias",model2.bout)
x_test=x_test[:100]


y_test=model2.predict(x_test)
preds = [np.argmax(p) for p in y_test]
print(preds)


    