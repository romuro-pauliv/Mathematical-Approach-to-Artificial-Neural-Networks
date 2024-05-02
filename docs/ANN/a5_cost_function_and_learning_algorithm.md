### Cost Function and Learning Algorithm

Based on the aforementioned truth, we can define a cost function $\psi: \mathbb{R}^n \rightarrow \mathbb{R}$ such that $\psi(w_0, w_1, \dots, w_n)$ where $w$ are the weights of layer $L$ of our artificial neural network. We will also define $Y = \varphi(Z^L)$ as the output of the layer and $\hat{Y}$ as the expected output of the layer. The result of the function $\psi$ is precisely the error between the layer output and the expected output. An example is given below:

\[
\psi_{MSE}(w_0, \dots, w_n) = \frac{1}{n}\sum_{i=0}^{n}(y_i(w_i) - \hat{y}_i)^2
\]

In $\psi_{MSE}$, we are using each element of $Y_{n \times 1}$ as $y_i$. Recapping the structure of $Y$ to clarify the situation of $y_i$:

\[
Y = \begin{bmatrix} y_0 \\ \vdots \\ y_n\end{bmatrix} = \begin{bmatrix} \varphi(z_0^L) \\ \vdots \\ \varphi(z_n^L)\end{bmatrix} = \begin{bmatrix} \varphi(w_0^L \cdot Z^{L-1}) \\ \vdots \\ \varphi(w_n^L \cdot Z^{L-1})\end{bmatrix}
\]

Thus, we can define $y_i$ as:

\[
y_i = \varphi(w_i^L \cdot Z^{L-1}), \quad w_i^L = \begin{bmatrix} w_0, w_1, \dots, w_h \end{bmatrix}, \quad Z^{L-1} = \begin{bmatrix} z_0 \\ z_1 \\ \vdots \\ z_h\end{bmatrix}
\]

Remember that when we use $w_i^L$ in $w_i^L \cdot Z^{L-1}$, we are referring to the transposed matrix $w^t$. In other cases, this is not applied. 


Based on what we learned in the Theoretical Learning Algorithm topic, for a function, \( f: \mathbb{R}^2 \rightarrow \mathbb{R} \), where \( f(x, y) = z \):

\[
\begin{bmatrix} x_{r+1} \\ y_{r+1}\end{bmatrix} := \begin{bmatrix} x_r \\ y_r \end{bmatrix} - \alpha \begin{bmatrix} \frac{\partial}{\partial x_r} f(x_r, y_r) \\\\ \frac{\partial}{\partial y_r} f(x_r, y_r) \end{bmatrix} \Rightarrow f(x_{r+1}, y_{r+1}) \leq f(x_r, y_r)
\]

Thus, for a single variable, in this case, \( x \), of the function \( f(x, y) \), to update it to satisfy the condition \( f(x_{r+1}, y_{r+1}) \leq f(x_r, y_r) \), we can define the following assignment:

\[
x_{r+1} := x_{r} - \alpha \frac{\partial}{\partial x_r} f(x_r, y_r)
\]

Based on this, we can apply the same concept to the cost function \( \psi_{MSE}(w_0^L, \dots, w_n^L) \) with respect to the weight vector \( w_i^L \). In the representation below, we define the value \( r \) to refer to the number of update iterations.

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \frac{\partial}{\partial w_{i[r]}^L} \psi_{MSE}(w_{0[r]}^L, \dots, w_{i[r]}^L)
\]

Throughout this topic, we will use the Mean Squared Error (MSE) cost function. This does not imply that only $\psi_{MSE}$ can be used; other types of cost functions can be applied, provided that all the reformulation below is redone.

Given the structure of $\psi_{MSE}$ and the supposed $y_i(w_i^L)$, we can define the following:

\[
\psi_{MSE}(w_{0[r]}^L, \dots, w_{n[r]}^L) = \frac{1}{n}\sum_{i=0}^{n}(\varphi(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i)^2
\]

Then for ${\psi}'_{MSE}(w_{0[r]}^L, \dots, w_{i[r]}^L)$, we have:

\[
\frac{\partial}{\partial w_{i[r]}^L} \psi_{MSE}(w_{0[r]}^L, \dots, w_{i[r]}^L) = \frac{1}{n}\frac{\partial}{\partial w_{i[r]}^L}(\varphi(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i)^2
\]

In the above definition, we notice that the partial derivative for each term of the sum different from $i$ is $0$, so we only have the term assigned with $i$ to solve. Let's define $E = \varphi(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i$, then we can apply the chain rule:

\[
\frac{\partial E^2}{\partial w_{i[r]}^L} = \frac{\partial E^2}{\partial E}\frac{\partial E}{\partial \varphi}\frac{\partial \varphi}{\partial w_{i[r]}^L}
\]

Solving the partial derivatives, we have:

\[
\frac{\partial E^2}{\partial w_{i[r]}^L} = 2E {\varphi}'(w_{i[r]}^L \cdot Z^{L-1}) \cdot Z^{L-1}
\]

Where the partial derivative of $u$ with respect to $\varphi$ is represented by the Lagrange notation ${\varphi}'$. With this, we can substitute $E$ into the resolution and again apply the definition of the weights $w_{i[r+1]}^L$:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \left ( \frac{2}{n} \left [\varphi(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i \right ] {\varphi}'(w_{i[r]}^L \cdot Z^{L-1}) \cdot Z^{L-1}\right )
\]

Knowing that $w_{i[r]}^L \in \mathbb{R}^{h \times 1}$ and also $Z^{L-1} \in \mathbb{R}^{h \times 1}$, we have:

\[
w_{i[r+1]}^L :=w_{i[r]}^L - \alpha \left ( \underbrace{\frac{2}{n} \left [\varphi(w_{i[r]}^L \cdot Z^{L-1}) - \hat{y}_i \right ] {\varphi}'(w_{i[r]}^L \cdot Z^{L-1})}_{\delta} \cdot Z^{L-1}\right )
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
