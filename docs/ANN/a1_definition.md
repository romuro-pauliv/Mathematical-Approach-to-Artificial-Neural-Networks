# Mathematical Approach to ANN - Part 1

> Developed by Pauliv, Rômulo. Data scientist, ML/AI Engineer

In the document below, we will explore the development of an artificial neural network $n$-$L$-$m$ ($n$ inputs, $L$ hidden layers, and $m$ outputs), as well as the theoretical learning algorithm based on the weights $w$ of each neuron involved.

In a artificial neural network (ANN) consisting of only one layer, commonly referred to as a perceptron, we have the following components that constitute a function $f(x) = y$, where $f: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}$:

### Inputs

The inputs will be denoted by the variable $x^t$, where $x^t = \begin{bmatrix}x_1 & x_2 & \dots & x_n\end{bmatrix}$, with each $x_i \in \mathbb{R}$ and for $x$ as a vector, we have $x \in \mathbb{R}^{n \times 1}$. The notation $x^t$ is based on the transposition of the vector $x$. The variable $x_0$ will be reserved to represent the bias in the future; therefore, the real $n$-inputs of the perceptron will always be referred to through $i = 1, \dots, n$.

### Weights

The weights will be responsible for modulating the $n$-inputs $x$. Thus, concerning the first layer, where the inputs will always be the vector $x$, we will always have $w^t = \begin{bmatrix} w_1 & w_2 & \dots & w_n \end{bmatrix}$, where each $w_i \in \mathbb{R}$, and for the vector $w$, we have $w \in \mathbb{R}^{n \times 1}$ with the quantity of $n$-inputs of $x$ equal to the $n$-weights of $w$. The weight $w_0$ will be reserved to represent the bias also. 

### Transfer Function $z(x)$

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

### Activation Function $\varphi(z)$
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
