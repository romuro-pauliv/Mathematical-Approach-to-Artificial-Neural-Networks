Considerando \( \varpi_i^L\) como os pesos do neurônio \( a_i^L \), onde \( \varpi_i^L \in \mathbb{R}^{h \times 1} \) e \( \varpi_i^L = [w_0^L, \dots, w_h^L] \), e \( A^{L-1} \in \mathbb{R}^{h \times 1}\) tal que \(A^{L-1} = [a_0^{L-1}, \dots, a_h^{L-1}]\), podemos estabelecer a relação para o neurônio \( a_i^L\):

\[
a_i^L = \varphi(\varpi_i^L \cdot A^{L-1}) \quad \therefore \quad \forall w_h^L \exists a_h^{L-1}
\]

Da mesma forma, para \( A^L \in \mathbb{R}^{n \times 1}\) onde \(A^L = [a_0^L, \dots, a_n^L]\), então \( \forall a_n^L \exists \varpi_n^L \). Assim, para a última camada \(L\), a saída é denominada \(A^L\), e podemos definir \(A^L = Y\) onde cada \(a_n^L = y_n\).

Portanto, tendo \(\hat{Y} \in \mathbb{R}^{n \times 1}\) como o valor esperado de \(Y\), podemos definir a função erro \(\psi: \mathbb{R^n} \rightarrow \mathbb{R}\):

\[
\psi(y_0, \dots, y_n) = \frac{1}{n} \sum_{i=0}^{n} (y_i - \hat{y}_i)^2
\]

Nosso objetivo é observar como os pesos \( \varpi_i^L\) influenciam o erro \( \psi(y_0, \dots, y_n)\), então:

\[
\frac{\partial}{\partial \varpi_i^L} \psi(y_0, \dots, y_n), \quad y_i= \varphi(z_i^L) = \varphi(\varpi_i^L \cdot A^{L-1})
\]

Resolvendo a derivada parcial para \( \varpi_i^L\) aplicando a regra da cadeia, obtemos:

\[
\frac{\partial}{\partial \varpi_i^L} \psi(y_0, \dots, y_n) = \frac{1}{n} \sum_{i=0}^{n}\frac{\partial E_i}{\partial y_i} \frac{\partial y_i}{\partial z_i^L} \frac{\partial z_i^L}{\partial \varpi_i^L}
\]

\[
\frac{\partial}{\partial \varpi_i^L} \psi(y_0, \dots, y_n) = \underbrace{\frac{2}{n}(y_i - \hat{y_i}) {\varphi}'(z_i^L)}_{\delta_i^L} \cdot A^{L-1}
\]

Portanto, pela definição acima, temos:

\[
\frac{\partial}{\partial \varpi_i^L} \psi(y_0, \dots, y_n) = \delta_i^L \cdot A^{L-1}
\]

Agora nosso objetivo é observar como os pesos \( \varpi^{l-1} \) influenciam o erro \( \psi(y_0, \dots y_n) \), então:

\[
\frac{\partial}{\partial w_h^{L-1}} \psi(y_0, \dots, y_n)
\]

Antes de resolver a derivada parcial acima, temos a definição de \(\varpi_h^{L-1}\):

Considerando \( \varpi_h^{L-1}\) como os pesos do neurônio \( a_h^{L-1} \), onde \( \varpi_h^{L-1} \in \mathbb{R}^{j \times 1} \) e \( \varpi_h^{L-1} = [w_0^{L-1}, \dots, w_j^{L-1}] \), e \( A^{L-2} \in \mathbb{R}^{j \times 1}\) tal que \(A^{L-2} = [a_0^{L-2}, \dots, a_j^{L-2}]\), podemos estabelecer a relação para o neurônio \( a_h^{L-1}\):

\[
a_h^{L-1} = \varphi(\varpi_h^{L-1} \cdot A^{L-2}) \quad \therefore \quad \forall w_j^{L-1} \exists a_j^{L-1}
\]

Então com base nas definições acima, temos que:

\[
Y = \begin{bmatrix} y_0 \\ \vdots \\ y_n\end{bmatrix} = \begin{bmatrix} \varphi(\varpi_0^L \cdot A^{L-1}) \\ \vdots \\ \varphi(\varpi_n^L \cdot A^{L-1})\end{bmatrix} \quad \therefore \quad A^{L-1} = \begin{bmatrix} a_0^{L-1} \\ \vdots \\ a_h^{L-1}\end{bmatrix} = \begin{bmatrix} \varphi(\varpi_0^{L-1} \cdot A^{L-2}) \\ \vdots \\ \varphi(\varpi_h^{L-1} \cdot A^{L-2}\end{bmatrix}
\]

Então, resolvendo a função custo com base em \( \varpi_h^{L-1} \) temos:

\[
\frac{\partial}{\partial \varpi_h^{L-1}} \psi(y_0, \dots, y_n) = \frac{1}{n} \sum_{i=0}^{n} \frac{\partial E_i}{\partial y_i} \frac{\partial y_i}{\partial z_i^L}\frac{\partial z_i^L}{\partial a_h^{L-1}} \frac{\partial a_h^{L-1}}{\partial z_h^{L-1}}\frac{\partial z_h^{L-1}}{\partial \varpi_h^{L-1}}
\]

Sabemos que todos os \(y_i\) onde \(i=0, \dots, n\) contém o elemento \(a_h^{L-1}\) devido sua dependencia através de \( y_i = \varphi(\varpi_i^L \cdot A^{L-1}) \) onde \(A^{L-1} = [a_0^{L-1}, \dots, a_h^{L-1}]\). Por isso todos os elementos do somatório que dependem estritamente de \(A^{L-1}\) serão deriváveis, mantendo o somatório.

Sabendo disso, podemos continuar a resolução onde podemos reutilizar o termo conhecido \( \delta_i^L \) para simplificar as duas primeiras derivadas parciais:

\[
\frac{\partial E_i}{\partial y_i} \frac{\partial y_i}{\partial z_i^L} = \delta_i^L \Rightarrow \sum_{i=0}^{n} \frac{\partial E_i}{\partial y_i} \frac{\partial y_i}{\partial z_i^L}\frac{\partial z_i^L}{\partial a_h^{L-1}} = \sum_{i=0}^{n} \delta_i^L \frac{\partial z_i^L}{\partial a_h^{L-1}}
\]

Para a derivada parcial de \(z_i^L\) com base em \( a_h^{L-1} \) podemos resolver da seguinte forma:

\[
\sum_{i=0}^{n} z_i^L = \sum_{i=0}^{n} \varpi_i^L \cdot A^{L-1}
\]

Logo, todos os elementos do somatório deverão ser derivados devido a dependência de \(A^{L-1}\). Para facilitar a resolução, vamos resolver para somente o índice \(n\) onde \(\varpi_n^L \cdot A^{L-1}\):

Sabemos que:

\[
\varpi_n^L \cdot A^{L-1} = \begin{bmatrix} w_0^L, \dots, w_h^L \end{bmatrix} \begin{bmatrix} a_0^{L-1} \\ \vdots \\ a_h^{L-1} \end{bmatrix} = \sum_{k=0}^{h} w_{k[n]}^L a_k^{L-1}
\]

Onde o índice \( [n] \) representa de qual vetor de neurônios estamos falando. Nesse caso, estamos se referindo ao vetor \( \varpi_n^L \). Logo para resolvermos a derivada parcial:

\[
\frac{\partial}{\partial a_h^{L-1}}\sum_{k=0}^{h} w_{k[n]}^L a_k^{L-1} = \frac{\partial w_{0[n]}^L a_0^{L-1}}{\partial a_h^{L-1}} + \dots + \frac{\partial w_{h[n]}^L a_h^{L-1}}{\partial a_h^{L-1}}
\]

Então, para todo o termo diferente de \(h\) o resultado será zero. Logo, teremos nosso resultado como:

\[
\frac{\partial z_i^L}{\partial a_h^{L-1}} = \frac{\partial}{\partial a_h^{L-1}}\sum_{k=0}^{h} w_{k[n]}^L a_k^{L-1} = w_{h[n]}^L
\]

Substituindo esse resultado no somatório proveniente da função custo temos:

\[
\sum_{i=0}^{n} \delta_i^L \frac{\partial z_i^L}{\partial a_h^{L-1}} = \sum_{i=0}^{n} \delta_i^L w_{h[i]}^L
\]

Recaptulando o objetivo principal e substituindo as resoluções já feitas:

\[
\frac{\partial}{\partial \varpi_h^{L-1}} \psi(y_0, \dots, y_n) = \frac{1}{n} \left ( \sum_{i=0}^{n} \delta_i^L w_{h[i]}^L \right ) \frac{\partial a_h^{L-1}}{\partial z_h^{L-1}}\frac{\partial z_h^{L-1}}{\partial \varpi_h^{L-1}}
\]

Agora para \(a_h^{L-1}\) onde \(a_h^{L-1} = \varphi(z_h^{L-1}) = \varphi(\varpi_h^{L-1} \cdot A^{L-2}) \) temos:

\[
\frac{\partial}{\partial \varpi_h^{L-1}} \psi(y_0, \dots, y_n) = \frac{1}{n} \left ( \sum_{i=0}^{n} \delta_i^L w_{h[i]}^L \right ) {\varphi}'(z_h^{L-1}) \cdot A^{L-2}
\]

---

### Definição geral

Para o ajuste dos pesos \(\varpi_h^{L-1}\) do neurônio \(a_h^{L-1}\) da camada oculta \(L-1\), onde sabemos que existe \(n\)-saídas da camada \(L\) representadas por \(y_0, \dots, y_n\) e para cada saída existe um vetor de neurônios \(\varpi_0^L, \dots, \varpi_n^L\) onde cada \(\varpi_n^L \in \mathbb{R}^{h \times 1}\) e \(\varpi_n^L = [w_0^L, \dots, w_h^L]\):

\[
\varpi_{h[r+1]}^{L-1} := \varpi_{h[r]}^{L-1} - \alpha  \left (  \frac{1}{n} \sum_{i=0}^{n} \delta_i^L w_{h[i]}^L \right ) {\varphi}'(z_h^{L-1}) \cdot A^{L-2}
\]
