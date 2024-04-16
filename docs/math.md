## Perceptron

In an Artificial Neural Network (ANN) comprised of a singular layer, $ x $ signifies the input, conventionally represented as $ x^t = (x_1, \dots, x_n) $, constituting a vector. This input undergoes processing within the neuron through weights $ w $, each indexed as $ w_i $ for $ i = 1, ..., n $, culminating in the generation of $ z $. This $ z $ value serves as the argument for the activation function $ \varphi $, thereby yielding $ y $ according to the relation: $ y = \varphi(z) $.

In this manner, $ z $ represents the outcome derived from the processing of $ n $ inputs $ x_i $ through respective weights $ w_i $.

```math
z = \begin{bmatrix} w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_1\\ x_2 \\ \vdots \\ x_n \end{bmatrix} + b = \sum_{i=1}^{n}x_iw_i + b = w^t \cdot x+b
```
In the aforementioned equation, we have defined the bias as $b$. We can substitute the bias relationship with $x_0 = 1$; thus, we obtain:

```math
z = \begin{bmatrix} w_0 = b & w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_0 = 1\\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \sum_{i=0}^{n}x_iw_i = w^t \cdot x
```

Thus, we delineate two perspectives: employing the bias as $ b $ in the function $ z = x \cdot w^t + b $ yields an affine function, while setting $ b = w_0 $ and $ x_0 = 1 $ engenders a linear function.

In the graphical representation below, we illustrate the processing of the $ n $-inputs $ x_i $ through the respective weights $ w_i $, resulting in $ z $. This is denoted as the _transfer function_.

<img src="/img/transfer_function.svg" alt="The perceptron"/>

Immediately following the definition of the _transfer function_, we encounter the activation function $ \varphi $, which is responsible for introducing nonlinearity into the process. Below, we present the two perspectives of the _activation function_ through the affine and linear function:

```math
\varphi(z) = \varphi(x \cdot w^t + b) = \varphi(\sum_{i=0}^{n}x_iw_i) = y
```
Below is the representation of a single operating artificial neuron.

<img src="/img/single_neuron.svg" alt="Single Functional Neuron"/>

Hence, we can define that $y$, in the perspective of the perceptron, we can define $F(x) = y$ such that $F: \mathbb{R}^m \rightarrow \mathbb{R}$ :

```math
F(x) = \varphi(w \cdot x + b) = y
```
Note that in the above representation of $ F(x) = y $, we did not specify the identification $ t $ of matrix transposition in the weight vector $ w $. We will omit this notation to avoid confusion with the layer index that we will introduce in the next topic regarding multilayer perceptron.


### Multilayer Perceptron (MLP)

For a multilayer perceptron structure, let's consider the previous process as $ y = F^1(x)=\varphi(w^1 \cdot x + b^1) $ where the index 1 under $ F^1(x) $ refers to the specific layer we are addressing. It is also important to note that when referring to a layer, we have the following representation:

<img src="/img/MLP_first_layer.svg" alt="Multilayer Perceptron First Layer"/>

Notice that in the above representation, we are referring solely to the first layer, which connects with the $ n $-inputs $ x $. For each layer $ L $, we have a variable quantity of $ i $ neurons. Therefore, we can represent the layer as follows:

```math
f^1(x) = \begin{bmatrix} w_{1_1} & w_{2_1} & \dots & w_{n_1} \\ w_{1_2} & w_{2_2} & \dots & w_{n_2} \\ \vdots & \vdots & \ddots & \vdots \\ w_{1_i} & w_{2_i} & \dots & w_{n_i} \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_i \end{bmatrix} = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_i \end{bmatrix}
```
Or through the linear approach:

```math
f^1(x) = \begin{bmatrix}w_{0_1}=b_1 & w_{1_1} & w_{2_1} & \dots & w_{n_1} \\w_{0_2}=b_2 & w_{1_2} & w_{2_2} & \dots & w_{n_2} \\ \vdots &  \vdots & \vdots & \ddots & \vdots \\ w_{0_i}=b_i & w_{1_i} & w_{2_i} & \dots & w_{n_i} \end{bmatrix} \cdot \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}  = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_i \end{bmatrix}
```
```math
f: \mathbb{R}^{n \times 1} \rightarrow \mathbb{R}^{i \times 1}\\
f^1(X_{n \times 1}) = W_{i \times n}^1 X_{n \times 1} = Z_{i \times 1}
```