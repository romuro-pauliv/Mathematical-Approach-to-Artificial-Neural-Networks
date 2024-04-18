## Perceptron
In a artificial neural network (ANN) consisting of only one layer, commonly referred to as a perceptron, we have the following components that constitute a function $f(x) = y$, where $f: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}$:

### Inputs

The inputs will be denoted by the variable $x^t$, where $x^t = \begin{bmatrix}x_1 & x_2 & \dots & x_n\end{bmatrix}$, with each $x_i \in \mathbb{R}$ and for $x$ as a vector, we have $x \in \mathbb{R}^{n \times 1}$. The notation $x^t$ is based on the transposition of the vector $x$. The variable $x_0$ will be reserved to represent the bias in the future; therefore, the real $n$-inputs of the perceptron will always be referred to through $i = 1, \dots, n$.

### Weights

The weights will be responsible for modulating the $n$-inputs $x$. Thus, concerning the first layer, where the inputs will always be the vector $x$, we will always have $w^t = \begin{bmatrix} w_1 & w_2 & \dots & w_n \end{bmatrix}$, where each $w_i \in \mathbb{R}$, and for the vector $w$, we have $w \in \mathbb{R}^{n \times 1}$ with the quantity of $n$-inputs of $x$ equal to the $n$-weights of $w$. The weight $w_0$ will be reserved to represent the bias also. 

### Transfer Function$z(x)$

Thus, $z(x)$ is responsible for relating the $n$-inputs with their respective $n$-weights to generate $z$, serving as the transfer function. As we are currently defining a layer, this being the first one, $z: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}$. To define the function $z(x)$, we can do so in two ways:

```math
z = \begin{bmatrix} w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_1\\ x_2 \\ \vdots \\ x_n \end{bmatrix} + b = \sum_{i=1}^{n}x_iw_i + b = w^t \cdot x+b
```
In the above definition, we have the _affine function_. However, we can also define the bias as $b = w_0$ and $x_0 = 1$ by substituting the bias $b$ in the affine function with a neuron whose weight serves as the bias, and its input is always set to 1. In this method, where the bias acts as a weight, we have a _linear function_.

```math
z = \begin{bmatrix} w_0 = b & w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_0 = 1\\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \sum_{i=0}^{n}x_iw_i = w^t \cdot x
```

In the graphical representation below, we illustrate the processing of the $n$-inputs $x_i$ through the respective weights $w_i$, resulting in $z$. This is denoted as the _transfer function_.

<img src="/img/transfer_function.svg" alt="The perceptron"/>

### Activation Function$\varphi(z)$
The activation function is responsible for introducing nonlinearity into the system. Therefore, it always receives the value of $z$. The activation function can be defined in various forms such as sigmoid, hyperbolic tangent, rectified linear unit, etc. In this topic, we will not focus on the behavior of each function yet but rather on how the system is structured with it. Thus, for only one layer of neurons with a single neuron, we have $\varphi: \mathbb{R} \rightarrow \mathbb{R}$:

```math
\varphi(z) = \varphi(w^t \cdot x) = \varphi(\sum_{i=0}^{n}x_iw_i) = y
```
Below is the representation of a single operating artificial neuron.

<img src="/img/single_neuron.svg" alt="Single Functional Neuron"/>

So, from the perspective of a perceptron, we can define the function $f(x) = y$, where $f: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}$:

```math
f(x) = \varphi(w^t \cdot x) = y
```

Note that in the above definitions, we used the linear function mode. Also, in the next topic, we will not use the notation $w^t$ indicating the transpose of the matrix. Therefore, it will be implicit that in the multiplication between layer inputs and their weights, the weight matrix will be transposed. In this case, we will see it as follows: $z(x) = w \cdot x$.


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