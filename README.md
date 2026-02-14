# model_from_scratch

Welcome to my **Machine Learning & Deep Learning Projects** repository!  
This repository showcases my journey from classical machine learning models to advanced deep learning architectures, including Transformers and Language Models (LM/LLM). Each project is documented with theory, implementation, results, and reflections.

## 🔹 Table of Contents

- [About](#about)  
- [Projects](#projects)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Technologies & Libraries](#technologies--libraries)  
- [Repository Structure](#repository-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## 🔹 About

This repository demonstrates practical experience with:

- **Classical ML:** Linear Regression, Logistic Regression, Decision Tree Regression, K-Means, TF-IDF  
- **Neural Networks:** Softmax Layer, Feedforward Neural Network, CNN, RNN  
- **Advanced Architectures:** Transformers, Language Models, LLMs (in progress)

The goal is to build a strong foundation, document each step, and provide reproducible code.



---
 
## 🔹 Projects
## 🔹 Projects

| Model / Project | Type | Description |
|-----------------|------|-------------|
| [Linear Regression](#linear-regression) | ML | Predict continuous targets using a linear approach ([Code](my_models.py)) |
| [Logistic Regression](#logistic-regression) | ML | Binary/multi-class classification ([Code](my_models.py)) |
| [TF-IDF (Term Frequency – Inverse Document Frequency)](#tf-idf) | NLP | Text vectorization for ML models ([Code](tfidf.py)) |
| [Decision Tree Models](#decision-tree-models) | ML/DL | Tree-based models for regression and classification ([Code](decision_tree_algorithm.py)) |
| [K-Means Clustering](#k-means-clustering) | ML | Unsupervised clustering algorithm ([Code](k_means.py)) |
| [Softmax Classifier](#softmax-classifier) | DL | Multi-class classification ([Code](softmax_regressor.py)) |
| [Feedforward Neural Network](#feedforward-neural-network) | DL | Fully connected neural network ([Code](neural_network.py)) |


 # Linear Regression 

This section documents the **Linear Regression** model implemented from scratch in Python.  

---

## 🔹 Objective

Predict **continuous target values** using a linear relationship between features and the output.  

---

## 🔹 Mathematical Foundation

The model assumes a linear relationship:

$$
y = A \cdot x + B
$$

Where:  
- \(A\) = slope of the line  
- \(B\) = intercept  

The **least squares method** is used to find the optimal parameters:

$$
A = \frac{ n \sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i \sum_{i=1}^{n} y_i }
         { n \sum_{i=1}^{n} x_i^2 - \left(\sum_{i=1}^{n} x_i \right)^2 }, 
\quad 
B = \frac{ \sum_{i=1}^{n} y_i - A \sum_{i=1}^{n} x_i }{n}
$$

**Evaluation Metrics:**

- **Mean Squared Error (MSE):**  

$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$


- **Root Mean Squared Error (RMSE):**  
  
$$
RMSE = \sqrt{MSE} 
$$
  

- **Mean Absolute Error (MEA):**  

$MEA = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$


---

## 🔹 Key Code Snippet

```python
class linear(model):
    def __init__(self, A=0, B=0):
        self.A = A
        self.B = B

    def fit(self, x, y):
        n = len(x)
        s, s1, s2, s3 = 0, 0, 0, 0
        for i in range(n):
            s += x[i] * y[i]
            s1 += x[i]
            s2 += y[i]
            s3 += x[i] ** 2
        self.A = np.round((n*s - s1*s2) / (n*s3 - s1**2), 4)
        self.B = (s2 - self.A*s1) / n
        print("Slope:", self.A, "Intercept:", self.B)

    def predict(self, x):
        return [self.A * xi + self.B for xi in x]

    def MSE(self, y_true, y_pred):
        return sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true)

    def RMSE(self, y_true, y_pred):
        return math.sqrt(self.MSE(y_true, y_pred))

    def MEA(self, y_true, y_pred):
        return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)
```


# Logistic Regression 

This section documents the **Logistic Regression** model implemented from scratch in Python.

---

## 🔹 Objective

Perform **binary classification** (0/1) or multi-class classification using **sigmoid** or **softmax** activation.  

---

## 🔹 Mathematical Foundation

**Sigmoid function** (for binary classification):

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Prediction probability**:

$ \hat{y} = \sigma(X \cdot W + B) $

Where:  
- $X$ = input features  
- $W$ = weights  
- $B$ = bias  

**Decision rule (for binary classification):**

$$
y_{\text{pred}} =
\begin{cases} 
1 & \text{if } \hat{y} \geq 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}
$$

**Gradient Descent Update**:

$ W := W - \alpha \frac{1}{n} X^T (\hat{y} - y) $

Where $\alpha$ is the learning rate.  

**Evaluation Metrics:**

- **Accuracy**:

$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$

- **Precision**:

$Precision = \frac{TP}{TP + FP}$

- **Recall**:

$Recall = \frac{TP}{TP + FN}$

- **F1 Score**:

$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

---

## 🔹 Key Code Snippet

```python
class logistic(model):
    def __init__(self, n_iter=1000, A=0, B=0, use=None):
        self.A = A
        self.B = B
        self.n_iter = n_iter
        self.use = use

    def fit(self, x, y, use="sigmoid"):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        x_bias = np.c_[np.ones(x.shape[0]), x]  # Add bias term
        w = np.zeros((x_bias.shape[1], 1))

        for i in range(self.n_iter):
            z = x_bias @ w
            h = sigmoid(z) if use=="sigmoid" else softmax(z)
            gradient = x_bias.T @ (h - y.reshape(-1, 1)) / y.size
            w = w - 0.01 * gradient

        self.A = w
        self.B = w[0]
        self.use = use
        return self.A, self.B

    def predict_proba(self, x):
        x = np.array(x, dtype=float)
        x_bias = np.c_[np.ones(x.shape[0]), x]
        z = x_bias @ self.A
        return sigmoid(z) if self.use=="sigmoid" else softmax(z)

    def predict(self, x, threshold=0.5):
        probs = self.predict_proba(x)
        if self.use=="sigmoid":
            return (probs >= threshold).astype(int).flatten()
        else:
            return np.argmax(probs, axis=1)
```
# TF-IDF (Term Frequency – Inverse Document Frequency)

This section documents the **TF-IDF** model implemented from scratch in Python for converting text into numerical features suitable for machine learning.

---

## 🔹 Objective

Transform textual data into a **numeric representation** that captures the importance of each word in a document relative to the entire corpus.

---

## 🔹 Mathematical Foundation

1. **Term Frequency (TF):**

The frequency of term $t$ in document $d$:

$$
TF_{t,d} = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d}
$$

2. **Inverse Document Frequency (IDF):**

Measures how rare a term is across all documents:

$$
IDF_t = \log \frac{N + 1}{1 + |\{d : t \in d\}|} + 1
$$

Where $N$ is the total number of documents.

3. **TF-IDF Score:**

$$
TFIDF_{t,d} = TF_{t,d} \times IDF_t
$$

---

## 🔹 Key Code Snippet

```python
import numpy as np
import re

def decompose(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()

class TFIDF:
    def __init__(self, vocab=None, idf=None):
        self.vocab = vocab
        self.idf = idf

    def compute_tf(self, data):
        tf = []
        doc_words = []
        for document in data:
            words = decompose(document)
            n = len(words)
            freq = {}
            for word in words:
                freq[word] = freq.get(word, 0) + 1
                if word not in doc_words:
                    doc_words.append(word)
            for word in freq:
                freq[word] /= n
            tf.append(freq)

        # Compute IDF
        idf = {}
        N = len(data)
        for word in doc_words:
            s = sum(1 for dec in tf if word in dec)
            idf[word] = np.log((N + 1) / (1 + s)) + 1

        # Compute TF-IDF
        tfidf = []
        for freq in tf:
            a = [freq.get(word, 0.0) * idf[word] for word in doc_words]
            tfidf.append(a)

        self.vocab = doc_words
        self.idf = idf
        return np.array(tfidf, dtype=float)

    def transform(self, data):
        if self.vocab is None or self.idf is None:
            print("You need to fit the model first")
            return
        tfidf = []
        for doc in data:
            words = decompose(doc)
            n = len(words)
            freq = {}
            for word in words:
                if word in self.vocab:
                    freq[word] = freq.get(word, 0) + 1
            for word in freq:
                freq[word] /= n
            a = [freq.get(word, 0.0) * self.idf[word] for word in self.vocab]
            tfidf.append(a)
        return np.array(tfidf, dtype=float)
```
#  Decision Tree Models

This section documents **Decision Trees** implemented from scratch in Python for both **classification** and **regression** tasks.

---

## 🔹 1. Decision Tree for Classification (Logistic Tree)

### Objective

Classify data into categories by recursively splitting the dataset based on **information gain (IG)**.

---

### Mathematical Foundation

1. **Entropy (Impurity Measure):**

For a set of labels $Y$:

$$
Entropy(Y) = - \sum_{c=0}^{C-1} p_c \log_2(p_c)
$$

Where $p_c$ is the proportion of class $c$ in the set.  

2. **Information Gain (IG):**

When splitting dataset $D$ into $D_\text{left}$ and $D_\text{right}$ on a feature:

$$
IG = Entropy(D) - \frac{|D_\text{left}|}{|D|} Entropy(D_\text{left}) - \frac{|D_\text{right}|}{|D|} Entropy(D_\text{right})
$$

The **feature and threshold with highest IG** is chosen for splitting.

--
``` python
def fit_tree(x,y):
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
```
# K-Means Clustering 

This section documents the **K-Means clustering algorithm** implemented from scratch in Python for **unsupervised learning** tasks.

---

## 🔹 Objective

Group data points into **k clusters** based on feature similarity by minimizing **within-cluster variance**.

---

## 🔹 Mathematical Foundation

1. **Cluster Assignment:**

Each data point $x_i$ is assigned to the nearest centroid $c_j$:

$$
label_i = \arg \min_{j \in \{1,...,k\}} \| x_i - c_j \|_2
$$

Where $\| \cdot \|_2$ is the Euclidean distance.

2. **Centroid Update:**

After assigning points, update each cluster centroid:

$$
c_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$

Where $C_j$ is the set of points assigned to cluster $j$.

3. **Objective Function (Within-Cluster Sum of Squares):**

$$
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - c_j \|_2^2
$$

K-Means iteratively **minimizes J** by alternating between assignment and centroid update.

---

## 🔹 Key Code Snippet

```python
import numpy as np

def k_means(x, k, max_iters=10):
    n_samples, n_features = x.shape
    # Randomly initialize centroids
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = x[idx]

    for _ in range(max_iters):
        labels = np.zeros(n_samples)

        # Assign points to nearest centroid
        for j in range(n_samples):
            distances = np.linalg.norm(x[j] - centroids[0])
            cluster = 0
            for c in range(1, k):
                dist = np.linalg.norm(x[j] - centroids[c])
                if dist < distances:
                    distances = dist
                    cluster = c
            labels[j] = cluster

        # Update centroids
        for c in range(k):
            points = x[labels == c]
            if len(points) > 0:
                centroids[c] = np.mean(points, axis=0)

    return labels, centroids
```
# Softmax Classifier 

This section documents the **Softmax classifier** implemented from scratch in Python for **multi-class classification** tasks.

---

## 🔹 Objective

Classify data into **multiple classes** by generalizing logistic regression using the **softmax function** and **cross-entropy loss**.

---

## 🔹 Mathematical Foundation

1. **Softmax Function:**

Converts raw scores (logits) into probabilities for each class:

$$
\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
$$

Where $C$ is the number of classes, and $z_j$ is the score for class $j$.

2. **Cross-Entropy Loss:**

Measures the difference between predicted probabilities and true labels:

$$
L = - \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

Where $y_{ij}$ is 1 if sample $i$ belongs to class $j$, else 0; $\hat{y}_{ij}$ is predicted probability.

3. **Gradient Descent Update:**

For each class $k$:

$$
w_k \gets w_k - \eta \, x_i \, (\hat{y}_{ik} - y_{ik}) 
$$

$$
b_k \gets b_k - \eta \, (\hat{y}_{ik} - y_{ik})
$$

Where $\eta$ is the learning rate.

---

## 🔹 Key Code Snippet

```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # stability trick
    return exp_z / np.sum(exp_z)

def fit_softmax(x, y, learning_rate=0.01, epochs=1000):
    x = np.array(x)
    y = np.array(y)
    n_samples, n_features = x.shape
    n_classes = len(np.unique(y))

    b = np.zeros(n_classes)
    w = np.random.randn(n_features, n_classes) * 0.01

    Y = np.eye(n_classes)[y]  # one-hot encoding

    for _ in range(epochs):
        for j in range(n_samples):
            z = np.array([x[j] @ w[:, k] + b[k] for k in range(n_classes)])
            p = softmax(z)

            for k in range(n_classes):
                if k == y[j]:
                    p[k] -= 1
                w[:, k] -= learning_rate * x[j] * p[k]
                b[k] -= learning_rate * p[k]

    return w, b
```
# Feedforward Neural Network 

This section documents a **fully-connected feedforward neural network** implemented from scratch in Python for **supervised learning** tasks.

---

## 🔹 Objective

Train a **multi-layer perceptron (MLP)** for **classification tasks**, supporting multiple hidden layers, using **sigmoid activation** and **softmax output**.

---

## 🔹 Mathematical Foundation

1. **Forward Pass:**

For layer $l$, the activations $a^{(l)}$ are:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

Where $\sigma(z)$ is the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

For the output layer, **softmax** is used to get class probabilities:

$$
\hat{y}_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
$$

2. **Loss Function (Cross-Entropy):**

$$
L = - \sum_{i=1}^{n} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

3. **Backpropagation:**

- Compute gradient of output:

$$
\delta^{(L)} = \hat{y} - y
$$

- For hidden layers $l$:

$$
\delta^{(l)} = (\delta^{(l+1)} W^{(l+1)T}) \odot \sigma'(z^{(l)})
$$

Where $\odot$ is element-wise multiplication and $\sigma'(z) = \sigma(z)(1-\sigma(z))$.

- Weight and bias updates:

$$
W^{(l)} \gets W^{(l)} - \eta \, a^{(l-1)T} \delta^{(l)}
$$

$$
b^{(l)} \gets b^{(l)} - \eta \, \delta^{(l)}
$$

---

## 🔹 Key Code Snippet

```python
import numpy as np
from softmax_regressor import softmax

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def neural_network(x, y, learning_rate=0.01, n_layer=2, n_neurons=[5,5], epochs=1000):
    x = np.array(x)
    y = np.array(y)
    n_samples, n_features = x.shape
    n_classes = len(np.unique(y))

    # Initialize weights and biases
    wights = []
    biases = []

    y_onehot = np.eye(n_classes)[y]

    for i in range(epochs):
        for t in range(n_samples):
            A = []
            Z = []

            # Forward pass
            for j in range(n_layer):
                if i==0 and t==0:
                    if j!=0:
                        n_features=n_neurons[j-1]
                    w=np.random.randn(n_features, n_neurons[j])*0.01
                    wights.append(w)
                    b=np.zeros(n_neurons[j])
                    biases.append(b)
                else:
                    w=wights[j]
                    b=biases[j]
                
                a_prev = x[t] if j==0 else A[-1]
                z = a_prev @ w + b
                a = sigmoid(z)
                Z.append(z)
                A.append(a)

            if i==0 and t==0:
                w_out = np.random.randn(n_neurons[-1], n_classes) * 0.01
                b_out = np.zeros(n_classes)

            # Output layer
            z_out = A[-1] @ w_out + b_out
            p = softmax(z_out)

            # Backpropagation
            dZ_out = p - y_onehot[t]
            dw_out = np.outer(A[-1], dZ_out)
            db_out = dZ_out
            dA = dZ_out @ w_out.T

            dw = [0]*n_layer
            db = [0]*n_layer

            for l in reversed(range(n_layer)):
                dZ = dA * sigmoid_derivative(Z[l])
                dw[l] = np.outer(x[t] if l==0 else A[l-1], dZ)
                db[l] = dZ
                dA = dZ @ wights[l].T

            # Update weights and biases
            for k in range(n_layer):
                wights[k] -= learning_rate * dw[k]
                biases[k] -= learning_rate * db[k]
            w_out -= learning_rate * dw_out
            b_out -= learning_rate * db_out

    return wights, biases, w_out, b_out
```










