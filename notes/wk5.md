# Machine Learning - Week 5 - Neural Networks

## Notation

* $L$ = total number of layers in the network
* $s_l$ = number of units (not counting bias unit, by convention) in layer $l$
* $K$ = number of output units/classes ( = $s_L$)
  * $K \ne 2$ as a single output node covers binary classification
* $ \big(h_\Theta(x)\big)_i$ is the $i$-th output of $h_\Theta \in \Bbb R^K$

## NN cost function

Cost function of logistic regression (revision):

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

For Neural Networks, it is:

$$ J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{p=1}^{s_l} \sum_{n=1}^{s_{l+1}} ( \Theta_{n,p}^{(l)})^2$$

The new sums:
* Before the square brackets:
  * Include all $K$ output nodes
* In the regularisation part, include all theta values:
  * Loop over each non-input layer
  * Loop over the $s_l$ inputs (_**in**cluding_ the bias)
  * Loop over the $s_l + 1$ outputs (_**ex**cluding_ the bias)

The double sum is the logistic regression costs for each output layer node.
The triple sum is the square of the network's individual $\Theta$ elements.

## Backpropagation algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, like gradient descent in logistic and linear regression.

To find $\min\limits_\Theta J(\Theta)$ we need to find the partial derivative of $J(\Theta)$: $\quad \dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$

$\delta_j^{(l)}$ is the "error" of node $j$ in layer $l$.

$\delta^{(L)} = a^{(L)} - y \quad = h_\Theta(x) - y$


Ignoring the regularisation term (see later), we can compute the partial derivative terms by multiplying our activation values and our error values for each training example $t$:
$$ \dfrac{\partial J(\Theta)}{\partial \Theta_{i,j}^{(l)}} = \frac{1}{m}\sum_{t=1}^m a_j^{(t)(l)} {\delta}_i^{(t)(l+1)}$$

### Pseudocode

![Forward propagation](wk5-fwd-prop.png)

Initialise $\Delta$ to all zeros (or any other random values): $\Delta^{(l)}_{i,j} := 0, \; \forall l, i, j$

$\forall t \in $ training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$:

1. Set $a^{(1)} = x^{(t)}$

2. Perform forward propagation to compute $a^{(l)} \forall l \in \{1 \dots L\}$

3. Compute the error in the output layer: $\delta^{(L)} = a^{(L)} - y^{(t)}$

4. Work back through the hidden layers, apportioning errors $\delta^{(L-1)}, \delta^{(L-2)},\dotsc,\delta^{(2)}$ based on the weights in the $\Theta^{(l+1)}$s.
  $\delta^{(l)} = \big((\Theta^{(l)})^T \delta^{(l+1)}\big) \circ g'(z^{(l)})$
  $\delta^{(l)} = \big((\Theta^{(l)})^T \delta^{(l+1)}\big) \circ a^{(l)} \circ (1 - a^{(l)})$
  * If any $\delta_0^{(l)}$s are calculated, they can be discarded as they are not needed in the derivative calculations (we know the bias nodes will always be $=1$).
  * The '$\circ$' represents the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices%29 ) or `.*` operator in Octave.
  * $\delta^{(1)} = 0$ as $a^{(1)} = x$
  * $g'(z^{(l)}) = a^{(l)} \circ (1 - a^{(l)}) =$ the [derivative of the logistic function](https://en.wikipedia.org/wiki/Logistic_function#Derivative ): ${\frac {d}{dx}}g(x) = g(x)(1-g(x))$
  * $z^{(l)} = \Theta^{(j-1)}a^{(j-1)}$  

5. Update $\Delta$:
  $\Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)}$
  $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$ (vectorised)

---------------------------------------------------------
After computing $\Delta$ based on all training examples, we compute $D$ as follows:

$$D^{(l)}_{i,j} := \begin{cases}
  \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right) \quad &\text{if $j \ne 0$} \\[2ex]
  \dfrac{1}{m}\Delta^{(l)}_{i,j} &\text{if $j = 0$} \\[2ex]
\end{cases}$$

While the formal proof is pretty complicated, take it on faith that:  
$$\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta) = D_{ij}^{(l)}$$

## Intuitive understanding of backprop

Recall the cost function for neural networks:

$$ J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{p=1}^{s_l} \sum_{n=1}^{s_{l+1}} ( \Theta_{n,p}^{(l)})^2$$

To make it easier, take a single training example, binary classification ($k=1$), and no regularisation ($\lambda=0$). Then:

$cost(t) =-y^{(t)} \ \log (h_\Theta (x^{(t)})) - (1 - y^{(t)})\ \log (1 - h_\Theta(x^{(t)}))$

The $\delta$ values show how much the cost function changes as $\partial z_j^{(l)}$ changes:

$\delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t)$

While [the errata](https://www.coursera.org/learn/machine-learning/resources/go98N ) says:
> This statement is not strictly correct, and is provided as an intuition for how the backpropagation process works.

Other sites (first link below) give this as unqualified truth.

Web links
* http://pandamatak.com/people/anand/771/html/node37.html - short, raster formulae
* http://bigtheta.io/2016/02/27/the-math-behind-backpropagation.html - derives some equations used in the above

## Implementation: unrolling parameters

`fminunc` takes and returns a vector, not a matrix. 

    numel(ones(10, 11)) == 110

To unroll into a vector:
    thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]


If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

    Theta1 = reshape(thetaVector(1:110),10,11)
    Theta2 = reshape(thetaVector(111:220),10,11)
    Theta3 = reshape(thetaVector(221:231),1,11)

or:

    sz_Theta1 = numel(Theta1);
    sz_Theta2 = numel(Theta2);
    sz_Theta3 = numel(Theta3);
    offset = 1;
    Theta1 = reshape(thetaVector(offset:offset+sz_Theta1),size(Theta1);
    offset = sz_Theta1;
    Theta2 = reshape(thetaVector(offset:offset+sz_Theta2),size(Theta2));
    offset += sz_Theta2;
    Theta3 = reshape(thetaVector(offset:offset+sz_Theta3),size(Theta3));

## Gradient checking

There can be subtle bugs with implementing backward propagation. Gradient checking is a way of ensuring that backprop is implemented correctly.

Estimate the gradient.

Choosing $\epsilon = 10^{-4}$ is a good value. Numerical issues may arise if it is too small.
One-sided difference:
$\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta)}{\epsilon}$

Two-sided difference (more accurate):

$\dfrac{\partial}{\partial\Theta}J(\Theta) \approx \dfrac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon}$

With multiple theta matrices, we can approximate the derivative with respect to $\Theta_j$ as follows:

$\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}$

Octave code:

    epsilon = 1e-4;
    for i = 1:size(theta),
      thetaPlus = theta;
      thetaPlus(i) += epsilon;
      thetaMinus = theta;
      thetaMinus(i) -= epsilon;
      gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
    end;





[//]: #speeling (check)

# Pomodoros
## Tuesday
* 
