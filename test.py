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


#test bert
from bert import bert
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
    "I dislike long debugging sessions",
    "I love playing football with my friends",
    "The weather today is sunny and warm",
    "She enjoys reading books in the library",
    "We are going to watch a movie tonight",
    "He likes to eat pizza for lunch",
    "Python is a popular programming language",
    "They visited the museum last weekend",
    "I am learning machine learning and AI",
    "My dog loves to run in the park",
    "The cat is sleeping on the sofa",
    "We are planning a trip to the mountains",
    "She bought a new pair of shoes yesterday",
    "The sun rises in the east and sets in the west",
    "He is studying computer science at university",
    "I enjoy listening to music while studying",
    "They are playing basketball in the gym",
    "The children are drawing pictures in class",
    "My favorite color is blue",
    "The train leaves at 9 o'clock every morning",
    "I need to finish my homework before dinner",
    "She is cooking pasta for dinner tonight",
    "We went hiking in the forest last weekend",
    "He is writing a book about history",
    "The flowers in the garden are blooming",
    "I like drinking coffee in the morning",
    "They are watching a football match on TV",
    "My brother is learning to play the guitar",
    "The movie was very exciting and fun",
    "I visited my grandparents last summer",
    "She is practicing yoga every day",
    "We are celebrating my friend's birthday",
    "He likes painting landscapes in his free time",
    "The car is parked in front of the house",
    "I am studying mathematics and physics",
    "The dog is barking at the mailman",
    "They are building a treehouse in the backyard",
    "She enjoys swimming in the ocean",
    "We are planning to go to the beach tomorrow",
    "He bought a new laptop for work",
    "I like to eat ice cream in the summer",
    "The children are playing in the playground",
    "She is reading a novel by her favorite author",
    "We went to a concert last night",
    "He is learning French and Spanish",
    "I enjoy hiking in the mountains",
    "They are watching a documentary on animals",
    "My cat likes to chase birds in the garden",
    "She is learning how to bake cakes",
    "We are organizing a charity event next month",
    "He is fixing the bike in the garage",
    "I love painting and drawing in my free time",
    "The sun is shining brightly in the sky",
]

tok=tokenizer()

tok.fit(text_array)
train_data = text_array * 100  
vec = tok.encode(train_data)
vec = tok.padding(vec, 12)
emb=embedding(tok.wordid,32)
emb.embedding_tran()

print(vec[0])
ber=bert(2,32,3,len(tok.wordid),emb)


ber.fit(vec,1e-2)   

predict_text = [
    "[MASK] love ai",                      # predict i"
    "Deep learning is [MASK]",            # predict missing word
    "Python is [MASK]",                    # predict "great" or similar
    "I enjoy coding in [MASK]",           # predict "Python"
    "Neural networks are [MASK]",         # predict "interesting" or similar
    "I hate [MASK] in my code",           # predict "bugs"
    "Machine learning is [MASK]",         # predict "amazing"
    "Training models is [MASK]",          # predict "rewarding"
    "Sometimes programming is [MASK]",    # predict "stressful"
    "I love [MASK] problems"              # predict "solving"
]
vec_pre=tok.encode(predict_text)
vec_pre=tok.padding(vec_pre, 12)

pred=ber.predict(vec_pre)
print(pred)
id2word = {v: k for k, v in tok.wordid.items()}
for p in pred:
    
    print(f"The word with ID {p} is: {id2word[p]}")
from gpt import gpt
gp=gpt(2,32,12,len(tok.wordid),emb)
gp.fit(vec,1e-2)
predict_text = [
    "i love ",                      # predict what comes after "I love"
    "Deep learning is ",            # predict missing word
    "Python is ",                    # predict "great" or similar
    "I enjoy coding in ",           # predict "Python"
    "Neural networks are ",         # predict "interesting" or similar
    "I hate  ",           # predict "bugs"
    "Machine learning is ",         # predict "amazing"
    "Training models is ",          # predict "rewarding"
    "Sometimes programming is ",    # predict "stressful"
    "I love solving"              # predict "problem"
]
predict_text=tok.encode(predict_text)
predict_text=tok.padding(predict_text,12)
out=gp.predict(predict_text)
for p in out:
    
    print(f"The word with ID {p} is: {id2word[p]}")

    



