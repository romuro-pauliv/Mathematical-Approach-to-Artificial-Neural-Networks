In an Artificial Neural Network (ANN) comprised of a singular layer, $ x $ signifies the input, conventionally represented as $ x^t = (x_1, \dots, x_n) $, constituting a vector. This input undergoes processing within the neuron through weights $ w $, each indexed as $ w_i $ for $ i = 1, ..., n $, culminating in the generation of $ z $. This $ z $ value serves as the argument for the activation function $ \varphi $, thereby yielding $ y $ according to the relation: $ y = \varphi(z) $.

In this manner, $ z $ represents the outcome derived from the processing of $ n $ inputs $ x_i $ through respective weights $ w_i $.

```math
z = \begin{bmatrix} w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_1\\ x_2 \\ \vdots \\ x_n \end{bmatrix} + b = \sum_{i=1}^{n}x_iw_i + b = w^tx+b
```
In the aforementioned equation, we have defined the bias as $b$. We can substitute the bias relationship with $x_0 = 1$; thus, we obtain:

```math
z = \begin{bmatrix} w_0 = b & w_1 & w_2 & \dots &  w_n \end{bmatrix} \cdot \begin{bmatrix} x_0 = 1\\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \sum_{i=0}^{n}x_iw_i = w^tx
```

<img src="/img/perceptron.svg" alt="The perceptron"/>