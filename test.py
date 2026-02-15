import  my_models
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from TFIDf import TFIDF

print(my_models.__file__)



print(hasattr(my_models, "linear"))    
print(hasattr(my_models, "logistic"))  
#testing linear regression model
X = np.random.randn(30000,1)
y = np.random.randn(30000)
model1=LinearRegression()
model2=my_models.linear()
start=time.time()
model1.fit(X,y)
end=time.time()
print("sklearn time: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)
#testing tfidf model
start=time.time()
model2.fit(X,y)
end=time.time()
print("custom model time: ",end-start)
text=["are you learning machine learning",
      "machine learning is fun",
      "I love coding in python",
      "python is a great programming language",
      "do you love deep learning",
      "i hate bugs in my code",
      "i hate information systems"]
labal=[0,1,1,1,0,0,0]
tfidf_sklearn=TfidfVectorizer()
start=time.time()
X_sklearn=tfidf_sklearn.fit_transform(text).toarray()
end=time.time()
print("sklearn TFIDF time: ",end-start)
print("sklearn TFIDF result:\n",X_sklearn)
start=time.time()
tfidf_custom=TFIDF()
X_custom=tfidf_custom.compute_tf(text)
end=time.time()
print("custom TFIDF time: ",end-start)
print("custom TFIDF result:\n",(X_custom))
start=time.time()
model1.fit(X_sklearn,labal)
end=time.time()
print("sklearn model time with TFIDF: ",end-start)
print("model coefficients: ",model1.coef_)
print("model intercept: ",model1.intercept_)

start=time.time()
model2.tfidf_fit(X_custom,labal)
end=time.time()
print("custom model time with TFIDF: ",end-start)
print("model coefficients: ",model2.A)
print("model intercept: ",model2.B)
#testing logistic regression model
model3=my_models.logistic(1000)
start=time.time()
model3.fit(X_custom,labal,"softmax")
end=time.time()
print("custom logistic model time with TFIDF: ",end-start)
print("model coefficients: ",model3.A)
print("model intercept: ",model3.B)
model4=LogisticRegression()
start=time.time()
model4.fit(X_sklearn,labal)
end=time.time()
print("sklearn logistic model time with TFIDF: ",end-start)
print("model coefficients: ",model4.coef_)
print("model intercept: ",model4.intercept_)
y_pred=model3.predict(X_custom)
print(y_pred)
y_pred=model4.predict_proba(X_sklearn)
print(y_pred)
text_pr=["i love machine learning and coding in python","python is fun"]
X_sklearn_pr=tfidf_sklearn.transform(text_pr).toarray()
print("sklearn TFIDF transform result:\n",X_sklearn_pr)
X_custom_pr=tfidf_custom.transform(text_pr)
print("custom TFIDF transform result:\n",X_custom_pr)
y_pred_pr=model3.predict(X_custom_pr)
print(y_pred_pr)
y_pred_pr=model4.predict(X_sklearn_pr)
print(y_pred_pr)
#test decision tree
tree_model=my_models.decision_tree()
X_dt = np.array([1,0,0,1,1,0,1,0])
y_dt = np.array([0,0,1,1,1,0,1,0])
X_dt=X_dt.reshape(-1,1)
tree_model.fit(X_dt,y_dt)
tree_model.print_tree()
for x in [0,1,1,0,1,0]:
    pred=tree_model.predict(tree_model.root,x)
    print(f"Prediction for input {x}: {pred}")    
model5=DecisionTreeClassifier()
model5.fit(X_dt.reshape(-1,1),y_dt)
for x in [0,1,1,0,1,0]:
      pred=model5.predict(np.array([[x]]))
      print(f"Sklearn Prediction for input {x}: {pred[0]}")
#test decision tree regression      
tree_reg=my_models.decision_tree_regression()
X_reg = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y_reg = np.array([1.5,1.7,3.2,3.8,5.1,5.9,7.3,7.8])
tree_reg.fit(X_reg,y_reg)
tree_reg.print_tree()
for x in [2.5,4.5,6.5,8.5]:
    pred=tree_reg.predict(tree_reg.root,x)
    print(f"Regression Prediction for input {x}: {pred}")               

sof_model=my_models.softmax_regression()
X_sof = np.array([[1,2],[1,0],[0,1],[0,0],[2,1],[2,2]])
y_sof = np.array([0,0,1,1,2,2])
sof_model.fit(X_sof,y_sof)
print("Softmax model weights:\n",sof_model.weights)
print("Softmax model bias:\n",sof_model.bias)
y_sof_pred=sof_model.predict(X_sof)
print("Softmax model predictions:\n",y_sof_pred)
#test neural network
nn_model=my_models.neural_network_model()
X_nn = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nn = np.array([0,1,1,0])
nn_model.fit(X_nn,y_nn,learning_rate=0.1,n_layer=2,n_neurons=[2,2],epochs=1000)
print("Neural Network model weights:\n",nn_model.weights)
print("Neural Network model biases:\n",nn_model.biases)
print("Neural Network output layer weights:\n",nn_model.w_out)
print("Neural Network output layer bias:\n",nn_model.b_out)
y_nn_pred=nn_model.predict(X_nn)
print("Neural Network model predictions:\n",y_nn_pred)
#test k-means
kmeans_model=my_models.k_means_model()
X_km = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
k=2
labels,centroids=kmeans_model.fit(X_km,4,max_iters=10)
print("K-means model labels:\n",labels)
print("K-means model centroids:\n",centroids)
y_km_pred=kmeans_model.predict(X_km)
print("K-means model predictions:\n",y_km_pred)
from rnn import tokenizer, embedding
from rnn import recurent
from rnn import layer
text_array=[
    "I love AI",
    "Deep learning is fun",
    "Hello world",
    "Python is great",
    "RNN is powerful",
    "I love deep learning"
]


labels=[1, 1, 0, 1, 0, 1]
#tokenization
tok=tokenizer()
tok.fit(text_array)
vec=tok.encode(text_array)
vec_padded = tok.padding(vec, 5)


print("tokenization",vec)
#creating rnn model
model=layer()
model.addembedding(tok.wordid,5)
model.addrecun(6)
#train
model.fit(vec_padded,labels)
print("wight",model.w_out)
print("bias",model.b_out)
text_pre=[
    "i love python",
    "i love machine learning"
]
vec_pre=tok.encode(text_pre)
vec_pre=tok.padding(vec_pre,5)
#predict
result=model.predict(vec_pre)
print(result)
text_array = [
    "I love AI",
    "Deep learning is fun",
    "Hello world",
    "Python is great",
    "RNN is powerful",
    "I love deep learning",
    "Machine learning is amazing",
    "I enjoy coding in Python",
    "Artificial intelligence is the future",
    "Neural networks are interesting",
    "I hate bugs in my code",
    "Debugging is frustrating",
    "Syntax errors are annoying",
    "Sometimes programming is stressful",
    "I dislike slow computers",
    "I love solving problems",
    "Data science is fascinating",
    "I enjoy learning new algorithms",
    "Training models is rewarding",
    "I hate runtime errors",
    "Optimization is challenging",
    "I like experimenting with models",
    "Python makes programming easier",
    "I am learning deep learning",
    "I dislike complicated setups",
    "I enjoy clean code",
    "Machine learning can be tricky",
    "I love AI research",
    "Sometimes training takes too long",
    "I like visualizing data",
    "RNNs can remember sequences",
    "I hate missing semicolons",
    "I enjoy writing functions",
    "I dislike long debugging sessions"
]

# Binary labels: 1 = positive / interested, 0 = negative / frustrated
labels = [
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 0,
    0, 1, 1, 1, 0,
    1, 0, 1, 0, 1,
    1, 0, 1, 0
]

tok2=tokenizer()
tok2.fit(text_array)
vec_arr=tok2.encode(text_array)
vec_arr=tok2.padding(vec_arr,7)
print("the padding",vec_arr)


model2=layer()
model2.addembedding(tok2.wordid,7)
model2.addrecun(5)
model.addrecun(7)

model2.fit(vec_arr,labels)
print("model2 weights",model2.w_out)
print("model2 biases",model2.b_out)
#cnn test 
#test cnn
from cnn import flatt,maxpool,layerc
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



model3=layerc()


model3.addconv(n_kernel=1, kernel_size=(3,3), input_shape=(4,4,1), stride=1)
model3.addmaxpool(pool_size=(2,2), stirde=2)
model3.addflatt()

model3.fit(x, y, learning_rate=0.01, epoches=5)

print("Training done!")
print("Output weights:", model3.wout)
print("Output bias:", model3.bout)
x2=[[[1, 3, 1, 2],
     [0, 1, 0, 0],
     [2, 1, 0, 1],
     [2, 0, 1, 0]]]
y2=model3.predict(x2)
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
model2=layerc()
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



