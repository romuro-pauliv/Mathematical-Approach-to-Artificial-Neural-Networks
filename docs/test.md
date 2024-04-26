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

Remember that when we use $w_i^L$ in $w_i^L \cdot \Z^{L-1}$, we are referring to the transposed matrix $w^t$. In other cases, this is not applied. With all variables defined, we can apply the adjustment of $w_i^L$ through $w_i^L := w_i^L - \alpha \nabla \psi_{MSE}(w_i^L)$. To better understand the assignment of new values to $w_i^L$, let's add a representative of the iterations, called $r$. Then we have:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \nabla \psi_{MSE}(w_{i[r]}^L)
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