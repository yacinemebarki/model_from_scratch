## 🔹 Recurrent Neural Network (RNN)

This section documents the implementation of a **vanilla Recurrent Neural Network (RNN)**
## 🔹 Tokenizer & Embedding Layer

This section documents the implementation of **text tokenization and embedding** from scratch in Python using NumPy.

🔹 **Objective**  
- Convert raw text into numerical sequences for machine learning and deep learning models.  
- Learn distributed vector representations of words (embeddings) that capture semantic similarity.  
- Prepare text sequences for sequential models such as RNNs.

🔹 **Mathematical Foundation**

1. **Tokenization**  
   The tokenizer maps each unique word in the corpus to a unique integer ID:  

   $\text{word} \rightarrow \text{integer ID}$

   - `fit(texts)`: Builds the vocabulary and assigns IDs.  
   - `encode(text)`: Converts a text sequence into a sequence of integers.  
   - `padding(sequence, \text{maxlen})`: Ensures all sequences have the same length by prepending zeros:  

   $\text{padded\_seq} = [0, 0, ..., \text{word\_IDs}]$

2. **Embedding Layer**  
   Each integer ID is mapped to a dense vector of dimension $d$:

   $\text{word\_ID} \rightarrow \mathbf{v} \in \mathbb{R}^{d}$

   - `embedding_tran()`: Initializes embedding vectors randomly:

     $\mathbf{v}_i \sim \text{Uniform}(0, 0.1)$

   - `forward(vec_text)`: Maps a sequence of word IDs to their corresponding vectors. Words with ID $0$ are mapped to a zero vector.

   The output of the embedding layer is a matrix:

   $\text{Embedding Output} \in \mathbb{R}^{L \times d}$

   Where $L$ = sequence length, $d$ = embedding dimension.

🔹 **Files**  
- Implementation: [`tok.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/rnn/tok.py)  
- Classes: `tokenizer`, `embedding`  
- Methods: `fit`, `encode`, `padding`, `embedding_tran`, `forward`  

🔹 **Key Methods**

- `tokenizer.fit(texts)`: Build vocabulary from corpus  
- `tokenizer.encode(texts)`: Encode text sequences into integer IDs  
- `tokenizer.padding(text_vec, maxlen)`: Pad sequences to uniform length  
- `embedding.embedding_tran()`: Initialize embedding vectors randomly  
- `embedding.forward(vec_text)`: Map integer sequences to embedding vectors

## 🔹 Recurrent Neural Network (RNN)

This section documents the implementation of a **simple RNN layer** from scratch in Python using NumPy.

🔹 **Objective**  
- Model sequential data (e.g., text, time series) by maintaining a hidden state that carries information across time steps.  
- Learn temporal dependencies in sequences for classification, prediction, or embedding tasks.

🔹 **Mathematical Foundation**  

1. **Forward Pass**  
At time step $t$, given input vector $\mathbf{x}_t \in \mathbb{R}^{n}$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^{H}$, the current hidden state $\mathbf{h}_t$ is computed as:

$$
\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W} + \mathbf{h}_{t-1} \mathbf{W}_h + \mathbf{b}_h)
$$

Where:  
- $\mathbf{W} \in \mathbb{R}^{n \times H}$ is the input-to-hidden weight matrix  
- $\mathbf{W}_h \in \mathbb{R}^{H \times H}$ is the hidden-to-hidden recurrent weight matrix  
- $\mathbf{b}_h \in \mathbb{R}^{H}$ is the bias vector  
- $\tanh$ is the activation function ensuring non-linearity  

2. **Backward Pass (Backpropagation Through Time, BPTT)**  
During training, the gradients are propagated backward through time:

$$
\delta \mathbf{h}_t = \delta \mathbf{h}^{\text{output}}_t + \delta \mathbf{h}_{t+1} 
$$

$$
\delta_{\text{tanh}} = \delta \mathbf{h}_t \odot (1 - \mathbf{h}_t^2)
$$

Weight updates (clipped for stability):

$$
\mathbf{W} \gets \mathbf{W} - \eta \, \delta_{\text{tanh}} \mathbf{x}_t^T
$$

$$
\mathbf{W}_h \gets \mathbf{W}_h - \eta \, \delta_{\text{tanh}} \mathbf{h}_{t-1}^T
$$

$$
\mathbf{b}_h \gets \mathbf{b}_h - \eta \, \delta_{\text{tanh}}
$$

Where $\eta$ is the learning rate, and $\odot$ denotes element-wise multiplication.

🔹 **Key Code Snippet**


- Implementation: [`recunn.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/rnn/recunn.py)  
- Classes: `recurent`
- Methods: `fit`, `backdrop`

## 🔹 RNN Pipeline with Tokenizer and Embedding

This section documents the implementation of a **text-processing RNN pipeline** from scratch in Python using NumPy.

🔹 **Objective**  
- Encode text sequences into embeddings.  
- Process sequences with a recurrent layer (RNN) to capture temporal dependencies.  
- Perform classification on the final hidden state.  

🔹 **Pipeline Overview**  
1. **Tokenizer**  
   - Converts raw text into integer sequences.  
   - Maintains a vocabulary mapping words to indices.  
   - Can pad sequences to a fixed length for batch processing.

2. **Embedding Layer**  
   - Maps integer word indices to dense vectors of size $d$.  
   - Each word $w$ has a learned embedding vector $\mathbf{v}_w \in \mathbb{R}^d$.  
   - Forward pass: $\mathbf{a}_t = \text{embedding}(\text{word\_id}_t)$  

3. **Recurrent Layer (RNN)**  
   - Hidden state update at time $t$:

$$
\mathbf{h}_t = \tanh(\mathbf{a}_t \mathbf{W} + \mathbf{h}_{t-1} \mathbf{W}_h + \mathbf{b}_h)
$$

   - Backpropagation through time (BPTT):

$$
\delta \mathbf{h}_t = \delta \mathbf{y}_t \mathbf{W}_{\text{out}}^T + \delta \mathbf{h}_{t+1} 
$$

$$
\delta_{\text{tanh}} = \delta \mathbf{h}_t \odot (1 - \mathbf{h}_t^2)
$$

$$
\mathbf{W} \gets \mathbf{W} - \eta \, \delta_{\text{tanh}} \mathbf{a}_t^T
$$

$$
\mathbf{W}_h \gets \mathbf{W}_h - \eta \, \delta_{\text{tanh}} \mathbf{h}_{t-1}^T
$$

$$
\mathbf{b}_h \gets \mathbf{b}_h - \eta \, \delta_{\text{tanh}}
$$

4. **Output Dense Layer**  
   - Final hidden state $\mathbf{h}_T$ is passed through a dense layer to produce logits for each class:

$$
\mathbf{z} = \mathbf{W}_{\text{out}} \mathbf{h}_T + \mathbf{b}_{\text{out}}
$$

   - Predicted probabilities via activation (tanh in this case):

$$
\mathbf{y}_{\text{pred}} = \tanh(\mathbf{z})
$$ 
- Implementation: [`rnn.py`](https://github.com/yacinemebarki/model_from_scratch/blob/main/rnn/rnn.py)  
- Classes: `layer`
- Methods: `addrecun`, `addembedding`,`fit`,`predict`


