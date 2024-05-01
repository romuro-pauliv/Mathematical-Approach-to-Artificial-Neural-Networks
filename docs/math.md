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

### Theoretical Learning Algorithm

Let's consider a cost function $E(W)$, where it represents the error value between the output of the layer and the expected value. The objective of this function is to provide a method that, by modifying the values of $W$, we can reduce the error indicated by the function $E(W)$ until it is the smallest possible value.

For this, we can apply the gradient of the function $E(W)$. Let's recall the gradient theorem below:

__Theorem 1__ (Gradient of a function). _Let $E: R^n \rightarrow R$ be a differenciable function in the neighbourhood of some points $W = \begin{bmatrix} w_0 & w_1 & \dots & w_i\end{bmatrix}$ where $i$ represents the number of neurons in this hypothetical layer. Then, the gradient of $E$ at $W$, denote by $\nabla E(W)$,_

1. _represents the slope of the tangent line to the function $E$ at the point $w$_;
2. _points in the direction in which the function $E$ most rapidly increases; thus, $-\nabla E$ indicates the direction of fastest decreasing_;
3. _is orthogonal to the level surfaces (generalization of the concept of a level curve for a function of two variables) of $E$, i.e., those of the form $E(W) = k$ for a constant $k$_

#### Example of $\nabla f$:

To facilitate understanding, let's define the function $f: \mathbb{R}^2 \rightarrow \mathbb{R}$ such that:

```math
f(x, y) = x^2 + y^2
```

If we visualize the function graphically, we have:

<div style="text-align:center;">
    <img src="/img/gradient_example_1.svg" alt="f(x, y) = x^2 + y^2"/>
</div>

Based on the function $f(x, y)$, we can define $-\nabla f(x, y)$ according to Theorem 1, item 2. Thus, we obtain a vector field with the following deduction:

```math
- \nabla f(x, y) = \begin{bmatrix} -\dfrac{\partial f(x, y)}{\partial x} = -2x \\\\ -\dfrac{\partial f(x, y)}{\partial y} = -2y \end{bmatrix}
```

To facilitate visualization, we can have a two-dimensional perspective $(x, y)$ of the function $f(x, y)$ where the coloration of the graph below will indicate the value of the $z$-axis. The intention is that the vectors point towards the minimum of the function, where the farther from the minimum, the greater the resultant of the vector, and the closer to the minimum, the smaller the resultant of the vector.

<div style="text-align:center;">
    <img src="/img/gradient_example_2.svg" alt="-\nabla f(x, y)"/>
</div>

Having the gradient descent $-\nabla f(x, y)$ at hand, we can define the algorithm responsible for updating the coordinates $x, y$ until they converge to the minimum of the function $f(x, y)$. Therefore, we can correct the values of $x$ and $y$ for $i$ iterations by adjusting the values based on the gradient descent.

In an empirical way, we can deduce that updating the values of $x_{i+1}$ and $y_{i+1}$ based on the gradient of $x_i$ and $y_i$ is given by:
```math
\begin{bmatrix} x_{i+1} \\ y_{i+1}\end{bmatrix} = \begin{bmatrix} x_{i} \\ y_{i}\end{bmatrix} - \alpha \begin{bmatrix} 2x_{i} \\ 2y_{i}\end{bmatrix}
```

The term $\alpha$ will be introduced later in the proof of this structure; for now, let's use it as a multiplier of the gradient, where the higher the $\alpha$, the greater the correction towards the minimum of the function, and the lower the $\alpha$, the smaller the correction towards the minimum of the function. This way, we can understand that:

```math
f(x_{i+1}, y_{i+1}) \leq f(x_{i}, y_{i}) 
```
Iterating $i$ times, we know that the condition $f(x, y) = 0$ will be true.

Below, we will see a simulation where we have coordinates $x, y$ far from the minimum of the function. We know that, based on the vector field, we will have a straight-line trajectory towards the minimum of the function. Because of this, we implement a random variable $\tau \in [-0.1, 0.1]$ to visualize how the gradient will perform. Therefore, the update relation of the coordinates will be:

```math
\begin{bmatrix} x_{i+1} \\ y_{i+1}\end{bmatrix} = \begin{bmatrix} x_{i} \\ y_{i}\end{bmatrix} - \alpha \begin{bmatrix} 2x_{i} \\ 2y_{i}\end{bmatrix} + \begin{bmatrix} \tau \\  \tau \end{bmatrix}
```
Based on the formulation above, we have the following simulation with various values of $\alpha$:

<div style="text-align:center;">
    <img src="/img/gradient_example_3.svg" alt="Gradient Descent Minimum Simulation"/>
</div>

Based on the simulation above, we see that when $\alpha$ is small, we take a small step towards the minimum of the function. When $\alpha$ is large, we take a large step towards the minimum of the function. In more complex functions, where there may be local minima, a large step may avoid them, while a small step may trap the algorithm in these minima.

We will address more analytical questions on this subject later. Now, we should focus on the proof of this structure.

#### Proof

Given $x_i \in \mathbb{R}$ and a differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, we can define the gradient $\nabla f(x_0, x_1, \dots, x_n)$. Let's represent $x_0, x_1, \dots, x_n$ as $x$ in the derivation below:

```math
\nabla f(x) = \left(\frac{\partial f(x)}{\partial x_0}, \frac{\partial f(x)}{\partial x_1}, \dots, \frac{\partial f(x)}{\partial x_n}\right) = ({f}'(x_0), {f}'(x_1), \dots, {f}'(x_n))
```

Based on the truth above, we can use the definition of limit:

```math
\nabla f(x) \approx \left(\lim_{h \rightarrow 0} \frac{f(x_0 + h) - f(x_0)}{h}, \lim_{h \rightarrow 0} \frac{f(x_1 + h) - f(x_1)}{h}, \dots, \lim_{h \rightarrow 0} \frac{f(x_n + h) - f(x_n)}{h}\right)
```

Forgetting the coordinate description (solving as a single-variable equation), we have:

```math
\nabla f(x) \approx \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h} \Rightarrow f(x + h) \approx f(x) + h\nabla f(x)
```

Let's consider the scalar factor $h$ to decrease rapidly, where $h = -\alpha \nabla f(x)$ for non-negative and small enough $(\alpha \rightarrow 0)$. Then:

```math
f(x - \alpha \nabla f(x)) \approx f(x) - \alpha (\nabla f(x))^2
```

We know that $(\nabla f(x))^2 \geq 0$ is strictly true. Thus, this confirms the following relationship:

\[
f(x) - \alpha (\nabla f(x))^2 \leq f(x)
\]

Therefore, knowing also that $f(x) - \alpha (\nabla f(x))^2 \approx f(x - \alpha \nabla f(x))$, we can update the term in the inequality:

\[
f(x - \alpha \nabla f(x)) \leq f(x)
\]

With the above relationship, we prove that using the term $x - \alpha \nabla f(x)$ induces its result to always be less than $f(x)$ itself. Based on this, we can define an algorithm that updates the value of $x$ for $i$ iterations:

```math
f(\underbrace{x - \alpha \nabla f(x)}_{x_1}) \leq f(\underbrace{\quad x \quad }_{x_0})
```

As a consequence, the sequence of updates of the minimum values for the function $f(x)$ with initial values $x_0$ is:

```math
x_{i+1} = x_{i} - \alpha_i \nabla f(x_i)
```
### Cost Function

Based on the aforementioned truth, we can define a cost function $\psi: \mathbb{R}^n \rightarrow \mathbb{R}$ such that $\psi(w_0, w_1, \dots, w_n)$ where $w$ are the weights of layer $L$ of our artificial neural network. We will also define $Y = \sigma(Z^L)$ as the output of the layer and $\hat{Y}$ as the expected output of the layer. The result of the function $\psi$ is precisely the error between the layer output and the expected output. An example is given below:

\[
\psi_{MSE} = \frac{1}{n}\sum_{i=0}^{n}(y_i - \hat{y}_i)^2
\]

In $\psi_{MSE}$, we are using each element of $Y_{n \times 1}$ as $y_i$. Recapping the structure of $Y$ to clarify the situation of $y_i$:

\[
Y = \begin{bmatrix} y_0 \\ \vdots \\ y_n\end{bmatrix} = \begin{bmatrix} \sigma(z_0^L) \\ \vdots \\ \sigma(z_n^L)\end{bmatrix} = \begin{bmatrix} \sigma(w_0^L \cdot Z^{L-1}) \\ \vdots \\ \sigma(w_n^L \cdot Z^{L-1})\end{bmatrix}
\]

Thus, we can define $y_i$ as:

\[
y_i = \sigma(w_i^L \cdot Z^{L-1}), \quad w_i^L = \begin{bmatrix} w_0, w_1, \dots, w_h \end{bmatrix}, \quad Z^{L-1} = \begin{bmatrix} z_0 \\ z_1 \\ \vdots \\ z_h\end{bmatrix}
\]

Remember that when we use $w_i^L$ in $w_i^L \cdot Z^{L-1}$, we are referring to the transposed matrix $w^t$. In other cases, this is not applied. With all variables defined, we can apply the adjustment of $w_i^L$ through $x_{i+1} = x - \alpha {f}'(x)$ demostrated in Proof of Theoretical Learning Algorithm. To better understand the assignment of new values to $w_i^L$, let's add a representative of the iterations, called $r$. Then we have:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \frac{\partial}{\partial w_{i[r]}^L} \psi_{MSE}(w_{0[r]}^L, \dots, w_{i[r]}^L)
\]

Throughout this topic, we will use the Mean Squared Error (MSE) cost function. This does not imply that only $\psi_{MSE}$ can be used; other types of cost functions can be applied, provided that all the reformulation below is redone.

Given the structure of $\psi_{MSE}$ and the supposed $y_i$, we can define the following:

\[
\psi_{MSE}(w_{0[r]}^L, \dots, w_{n[r]}^L) = \frac{1}{n}\sum_{i=0}^{n}(\sigma(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i)^2
\]

Then for $\nabla \psi_{MSE}(w_{i[r]}^L)$, we have:

\[
\nabla \psi_{MSE}(w_{i[r]}^L) = \frac{1}{n}\frac{\partial}{\partial w_{i[r]}^L}(\sigma(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i)^2
\]

In the above definition, we notice that the partial derivative for each term of the sum different from $i$ is $0$, so we only have the term assigned with $i$ to solve. Let's define $u = \sigma(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i$, then we can apply the chain rule:

\[
\frac{\partial u^2}{\partial w_{i[r]}^L} = \frac{\partial u^2}{\partial u}\frac{\partial u}{\partial \sigma}\frac{\partial \sigma}{\partial w_{i[r]}^L}
\]

Solving the partial derivatives, we have:

\[
\frac{\partial u^2}{\partial w_{i[r]}^L} = 2u {\sigma}'(w_{i[r]}^L \cdot Z^{L-1}) \cdot Z^{L-1}
\]

Where the partial derivative of $u$ with respect to $\sigma$ is represented by the Lagrange notation ${\sigma}'$. With this, we can substitute $u$ into the resolution and again apply the definition of the weights $w_{i[r+1]}^L$:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \left ( \frac{2}{n} \left [\sigma(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i \right ] {\sigma}'(w_{i[r]}^L \cdot Z^{L-1}) \cdot Z^{L-1}\right )
\]

Knowing that $w_{i[r]}^L \in \mathbb{R}^{h \times 1}$ and also $Z^{L-1} \in \mathbb{R}^{h \times 1}$, we have:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \left ( \underbrace{\frac{2}{n} \left [\sigma(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i \right ] {\sigma}'(w_{i[r]}^L \cdot Z^{L-1})}_{\delta} \cdot Z^{L-1}\right )
\]

So, in a more simplified manner:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \delta_i^L \cdot Z^{L-1}
\]

which can also be represented as:

\[
\begin{bmatrix} w_{i0[r+1]}^L \\ \vdots \\ w_{ih[r+1]}^L \end{bmatrix} := \begin{bmatrix} w_{i0[r]}^L \\ \vdots \\ w_{ih[r]}^L \end{bmatrix} - \begin{bmatrix} \alpha \delta_0^L z_0^{L-1} \\ \vdots \\ \alpha \delta_h^L z_h^{L-1}\end{bmatrix}
\]

So, for each weight $w_{ih} \in \mathbb{R}$ composing the weight vector $w_i \in \mathbb{R}^{h \times 1}$, its updated value is as follows:

\[
w_{ih[r+1]}^L := w_{ih[r]}^L - \alpha \delta_h^L z_{h}^{L-1}
\]

### Backpropagation

This process is only valid for the outputs $\sigma(Z^L) = Y$. Therefore, for the hidden layers, the same process will be applied, but with a different approach to the cost function due to the absence of targets $\hat{Y}$ for the previous layers. This implies the development of the algorithm commonly known as backpropagation, which will be introduced in __Part 2__ of this document.

# References
1. García Cabello, J. Mathematical Neural Networks. Axioms 2022, 11, 80
2. Popoviciu, N.; Baicu, F. The Mathematical Foundation and a Step by Step Description for 17 Algorithms on Artificial Neural Networks. In Proceedings of the 9th WSEAS International Conference on AI Knowledge Engineering and Data Bases, Cambridge UK, 20–22 February 2010; p. 23.
3. Leshno, M.; Lin, V.Y.; Pinkus, A.; Schocken, S. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Netw. 1993, 6, 861–867
4. Cooper, A.M.; Kastner, J.; Urban, A. Efficient training of ANN potentials by including atomic forces via Taylor expansion and application to water and a transition-metal oxide. Comput. Mater. 2020, 6, 54.