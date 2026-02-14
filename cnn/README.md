## 🔹 Convolutional Neural Network (CNN)

This section demonstrates the implementation of a **Convolutional Layer from scratch** in Python.

🔹 **Objective**  
Extract spatial features from images or feature maps using learnable kernels and ReLU activation. It supports multi-channel inputs (e.g., RGB images) and stride-based convolutions, forming the foundation for deeper CNN architectures.

🔹 **Mathematical Foundation**  
For an input patch $X$ and kernel $K$ with bias $b$, the convolution operation is:

$$
Z[i,j] = \sum_{c=1}^{C} \sum_{u=1}^{k_h} \sum_{v=1}^{k_w} X[i \cdot s + u, j \cdot s + v, c] \cdot K[u,v,c] + b
$$

where:  
- $k_h, k_w$ = kernel height and width  
- $C$ = number of input channels  
- $s$ = stride  

The activation function applied after convolution is **ReLU**:

$$
A[i,j] = \text{ReLU}(Z[i,j]) = \max(0, Z[i,j])
$$

During backpropagation, gradients are computed as:

$$
\frac{\partial \mathcal{L}}{\partial K} = \sum_{i,j} \text{dout}[i,j] \cdot X_{\text{patch}}
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{i,j} \text{dout}[i,j]
$$

$$
\frac{\partial \mathcal{L}}{\partial X} = \sum_{k} K_k * \text{dout}_k
$$

🔹 **Files**  
- Implementation: [`conv_layer.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/cnn/conv.py)  
- Classes: `kernel`, `conv`  
- Functions: `relu`, `relu_derivative`  

🔹 **Key Methods**  
- `conv.forward(input)`: Compute convolution + ReLU activation  
- `conv.backdrop(dout, lr)`: Backpropagate gradients and update kernel weights and bias

## 🔹 Max Pooling Layer

This section documents the **Max Pooling layer** implemented from scratch in Python.

🔹 **Objective**  
Reduce the spatial dimensions of feature maps while retaining the most salient features. Max pooling is commonly used in CNN architectures to downsample inputs and reduce computation.

🔹 **Mathematical Foundation**  
For an input feature map $X$ with channels $C$ and a pooling window of size $(p_h, p_w)$ with stride $s$, the output $Y$ is computed as:

$$
Y[i,j,c] = \max \big( X[i \cdot s : i \cdot s + p_h, \; j \cdot s : j \cdot s + p_w, \; c] \big)
$$

During backpropagation, the gradient is propagated **only to the maximum value in each patch**:

$$
\frac{\partial \mathcal{L}}{\partial X[i,j,c]} =
\begin{cases} 
\dout[i',j',c] & \text{if } X[i,j,c] = \max(\text{patch}) \\
0 & \text{otherwise} 
\end{cases}
$$

Where $\dout[i',j',c]$ is the gradient from the next layer corresponding to the pooled output.

🔹 **Files**  
- Implementation: [`maxpool_layer.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/cnn/maxpool.py)  
- Class: `maxpool`  
- Methods: `forward(input)`, `backdrop(dout, lr)`

🔹 **Key Methods** 

- `maxpool.forward(input)`: Compute max pooling on the input feature map

- `maxpool.backdrop(dout, lr)`: Backpropagate gradients through the pooling operation


## 🔹 Flatten Layer

This section documents the **Flatten layer** implemented from scratch in Python.

🔹 **Objective**  
Transform a multi-dimensional input (e.g., feature maps from a CNN) into a one-dimensional vector suitable for fully connected layers in neural networks.

🔹 **Mathematical Foundation**  
Given an input tensor $X$ of shape $(H, W, C)$, the Flatten layer reshapes it into a vector $Y$ of size:

$$
Y = \text{vec}(X) \quad \text{with size } H \cdot W \cdot C
$$

During backpropagation, the gradient with respect to the input is reshaped back to its original dimensions:

$$
\frac{\partial \mathcal{L}}{\partial X} = \text{reshape} \left( \frac{\partial \mathcal{L}}{\partial Y}, \text{shape}(X) \right)
$$

🔹 **Files**  
- Implementation: [`flatten_layer.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/cnn/flatt.py)  
- Class: `flatt`  
- Methods: `forward(input)`, `backdrop(da, lr)`

🔹 **Key Methods**  
- `flatt.forward(input)`: Flatten the input tensor to a 1D vector  
- `flatt.backdrop(da, lr)`: Reshape the gradient vector back to the original input shape

## 🔹 Convolutional Neural Network (CNN) with Dense Layer

This section documents a CNN architecture implemented from scratch in Python, including **Conv, MaxPool, Flatten, and Dense layers**.

🔹 **Objective**  
- Extract hierarchical features from multi-dimensional input (e.g., images) using convolutional and pooling layers.  
- Transform features into a vector using a Flatten layer.  
- Perform classification with a fully connected (Dense) layer using softmax activation.

🔹 **Mathematical Foundation**

1. **Convolution Layer**  
   Given an input tensor $X$ and a kernel $K$ of size $(k_h, k_w, c)$:

   $$
   Z[i,j,k] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c=0}^{C-1} X[i+m, j+n, c] \cdot K[m,n,c] + b_k
   $$

   Activation (ReLU):

   $$
   A[i,j,k] = \text{ReLU}(Z[i,j,k]) = \max(0, Z[i,j,k])
   $$

   Backpropagation for convolution:

   $$
   dK = \sum_{i,j} \text{patch} \cdot dZ[i,j,k], \quad db = \sum_{i,j} dZ[i,j,k]
   $$

2. **Max Pooling Layer**  
   Reduces spatial dimensions by taking the maximum in a sliding window:

   $$
   Y[i,j,c] = \max_{m,n} X[i \cdot s + m, j \cdot s + n, c]
   $$

   Backpropagation:

   $$
   dX[i \cdot s + m^*, j \cdot s + n^*, c] = dY[i,j,c]
   $$

   where $(m^*, n^*)$ is the position of the max in the pooling window.

3. **Flatten Layer**  
   Converts multi-dimensional feature maps into a 1D vector:

   $$
   Y = \text{vec}(X), \quad |Y| = H \cdot W \cdot C
   $$

   Gradient during backpropagation:

   $$
   \frac{\partial \mathcal{L}}{\partial X} = \text{reshape}\left(\frac{\partial \mathcal{L}}{\partial Y}, \text{shape}(X)\right)
   $$

4. **Dense Layer (Fully Connected)**  
   Takes the flattened vector $a$ and produces logits $z$:

   $$
   z = a \cdot W + b
   $$

   Softmax activation for multi-class output:

   $$
   \hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
   $$

   Loss gradient for backpropagation:

   $$
   dW = a^T \cdot (\hat{y} - y), \quad db = \hat{y} - y, \quad da = (\hat{y} - y) \cdot W^T
   $$

🔹 **Files**  
- Implementation: [`cnn_layers.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/cnn/convnn.py)  
- Class: `layer`  
- Methods: `addconv`, `addmaxpool`, `addflatt`, `fit`, `predict`  

🔹 **Key Methods**
- `addconv(n_kernel, kernel_size, input_shape, stride)`: Add a convolutional layer  
- `addmaxpool(pool_size, stride)`: Add a max pooling layer  
- `addflatt()`: Add a flatten layer to convert feature maps to a vector  
- `fit(x, y, learning_rate, epoches)`: Train the CNN with backpropagation  
- `predict(x)`: Make predictions with the trained CNN
  

  
  

  

