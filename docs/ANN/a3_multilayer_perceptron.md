### Multilayer Perceptron (MLP)

Based on the general function from the previous topic, we know that $Z^1 = W^1 \cdot X$, where $X$ refers to the $n$-inputs, $W^1$ represents the weights of each $i$-neuron in the first layer, and $Z^1$ denotes the output vector of the transfer function of the first layer. With this understanding, we can apply $L$ more layers to the system, where the input of layer $L$ is the output vector of layer $L-1$:

```math
Z^L = W^L \cdot Z^{L-1} \quad L=1, \dots, L
```

With this in mind, when $L-1 = 0$, we understand that we are referring to the input layer $X$. The representation below will be clearer:


<img src="/img/input_MLP.svg" alt="MLP Input and Hidden Layers"/>

Therefore, with this structure in place, it becomes clear that in the last layer $L$, we can apply the activation function $\varphi(Z)$ to obtain the output vector $Y$:

```math
Y = \varphi(Z^L) = \varphi(W^L \cdot Z^{L-1})
```

__Definition 1__: _A Multilayer perceptron is a function_ $F: \mathbb{R}^n \rightarrow \mathbb{R}^m$. _It is an $n$-$L$-$m$ perceptron ($n$-inputs, $L$ hidden layers, and $m$ outputs) if it is a function of the form_

```math
F(X) = \varphi(f^L(f^{L-1}(\dots f^1(X)))), \quad \text{for} \quad X^t=\begin{bmatrix} x_0 & x_1 & \dots & x_n \end{bmatrix}
```

_Where \( f^1(X) = W^1 \cdot X \) and for the remaining layers \( f^i(Z^{i-1}) = W^i \cdot Z^{i-1} \)_.

Notice that the usability of the activation function may vary. For example, in some problems, we may use the activation function at the output of each layer, while in others, we may not. For regression tasks in ANN, we won't use the activation layer in $f^L(Z^{L-1})$, where as for classification problems, we may utilize the activation function in layer $L$.

Based on this, we have the freedom to create structures depending on the problem at hand. We will explore the application of this further ahead.