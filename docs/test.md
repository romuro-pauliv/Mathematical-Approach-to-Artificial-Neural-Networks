### Cost Function

Tendo como base a verdade acima, podemos definir uma função custo $\psi: \mathbb{R}^n \rightarrow \mathbb{R}$ tal que $\psi(w_0, w_1, \dots, w_n)$ onde $w$ são os pesos da camada $L$ da nossa rede neural aritifical. Também iremos definir $Y = \sigma(Z^L)$ como o output da camada e $\hat{Y}$ como a saída esperada da camada. O resultado da função $\psi$ é propriamente o erro entre o output da camada e a saída esperada. Um exemplo abaixo:

```math
\psi_{MSE} = \frac{1}{n}\sum_{i=0}^{n}(y_i - \hat{y}_i)^2
```

Em $\psi_{MSE}$ estamos utilizando cada elemento de $Y_{n \times 1}$ como $y_i$. Recaptulando a estrutura de $Y$ para elucidar a situação de $y_i$:

```math
Y = \begin{bmatrix} y_0 \\ \vdots \\ y_n\end{bmatrix} = \begin{bmatrix} \sigma(z_0^L) \\ \vdots \\ \sigma(z_n^L)\end{bmatrix} = \begin{bmatrix} \sigma(w_0^L \cdot Z^{L-1}) \\ \vdots \\ \sigma(w_n^L \cdot Z^{L-1})\end{bmatrix}
```

Logo, podemos definir $y_i$ como:

```math
y_i = \sigma(w_i^L \cdot Z^{L-1}), \quad w_i^L = \begin{bmatrix} w_0, w_1, \dots, w_h \end{bmatrix}, \quad Z^{L-1} = \begin{bmatrix} z_0 \\ z_1 \\ \vdots \\ z_h\end{bmatrix}
```

