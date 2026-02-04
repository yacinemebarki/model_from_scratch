import  my_models
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from TFIDf import TFIDF
from connn import layer, kernel_conv,relu,relu_derivative,cnn,cnn_predict
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
#test cnn 
model=layer()   
model.add_conv(n_kernel=2,kernel_size=(3,3),stride=1,input_shape=(5,5,1))
model.add_maxpool(pool_size=(2,2),stride=2)
model.add_flatten(n_neurons=3)
x_cnn = np.random.randn(1,5,5,1)
y_cnn = np.array([0])
cnn_weights,cnn_biases,fc_weights,fc_biases=cnn(x_cnn,y_cnn,model,learning_rate=0.01,epochs=10)
print("CNN Convolutional layer weights:\n",cnn_weights)
print("CNN Convolutional layer biases:\n",cnn_biases)
print("CNN Fully connected layer weights:\n",fc_weights)
print("CNN Fully connected layer biases:\n",fc_biases)
y_cnn_pred=cnn_predict(x_cnn,model,cnn_weights,cnn_biases,fc_weights,fc_biases)
print("CNN model predictions:\n",y_cnn_pred)



