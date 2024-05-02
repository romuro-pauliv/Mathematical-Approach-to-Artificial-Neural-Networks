### Layer with Multiple Neurons

In the previous definition, we had a layer with only one neuron. In this topic, we will define the behavior of a layer with $i$-neurons. Thus, the function $f(x) = y$ will not define only the behavior of one neuron but of $i$-neurons.

Therefore, we will have the definition of $f^L(x)$ where $L$ is the index of the layer. Each layer $L$ can have $i$ neurons. So, for each neuron $i$, we will have an output of the transfer function $z_{i}^1$ in the layer $f^1(x)$.

<img src="/img/layer_with_multiple_neurons.svg" alt="Multilayer Perceptron First Layer"/>

As an example, let's analyze $z_{i}^1 = w_{i}^1 \cdot x$. For $w_{i}^1$, which represents the weights of neuron $i$ in layer 1, we have:

```math
w_{i}^1 = \begin{bmatrix} w_{0_i} & w_{1_i} & \dots & w_{n_i}\end{bmatrix}, \forall w_{n_i}\exists x_n
```
Therefore, for input into the layer, there will always be a weight $w$. However, this does not imply that there should be $i$ neurons for each $n$-inputs. So, for the function $z(x)$, where $z^1: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}^{i \times 1}$, we have:

```math
z^1(x) = \begin{bmatrix}w_{0_1}=b_1 & w_{1_1} & w_{2_1} & \dots & w_{n_1} \\w_{0_2}=b_2 & w_{1_2} & w_{2_2} & \dots & w_{n_2} \\ \vdots &  \vdots & \vdots & \ddots & \vdots \\ w_{0_i}=b_i & w_{1_i} & w_{2_i} & \dots & w_{n_i} \end{bmatrix} \cdot \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}  = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_i \end{bmatrix}
```

We can also represent the transfer function in the following manner. Note that after defining the function below, we denote $Z^1$ in uppercase for the output vector of layer and $W_{i}$ for the weights of neuron $i$, facilitating comprehension:

```math
Z^1 =\begin{bmatrix} z_1 = \sum_{k=0}^{n} w_{k_1}x_k  \\ z_2 = \sum_{k=0}^{n} w_{k_2}x_k  \\ \vdots \\ z_i = \sum_{k=0}^{n} w_{k_i}x_k  \end{bmatrix} = W^1 \cdot X
```
Applying now the activation function $\varphi$, which will generate $f^1(X) = Y$ where $f^1: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}^{i \times 1}$, we have:

```math
f^1(X) = \varphi(Z^1) = \varphi(W^1 \cdot X)
```
Now, demonstrating the dimensions of each matrix, we have:

```math
f^1(X_{n \times 1}) = \varphi(W^1_{i \times n}X_{n \times 1}) = Y_{i \times 1}
```
Where $n$ is the quantity of $n$-inputs in the vector $X$, and $i$ is the quantity of neurons in layer 1. Thus, we clearly see that $n \neq i$ is possible, where the output of the layer will always be of size $i$.