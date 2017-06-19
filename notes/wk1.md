# Coursera Machine Learning (Stanford, Andrew Ng) - Week 1

## Course resources
* [CS299 materials](http://cs229.stanford.edu/materials.html )

* [Course FAQ](https://www.coursera.org/learn/machine-learning/supplement/gBboB/frequently-asked-questions )

* [Programming test cases and tutorials](
https://www.coursera.org/learn/machine-learning/discussions/all/threads/VfMe7CRmEeebrBIWZEFM5A )

## Resources (other)
* [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html )
Online textbook. Quite accessible, focussing on core concepts and principles. Has a visual example of neural nets computing any function.

* [Dive into Machine Learning (GitHub)](
https://github.com/hangtwenty/dive-into-machine-learning )

* [Unsupervised Feature Learning and Deep Learning Tutorial (Stanford)](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial )

* [Holehouse lecture notes](http://www.holehouse.org/mlclass/ )
* [Chris McCormick's notes (search "02 Apr 2013")](http://mccormickml.com/archive/ )

### Maths
* [Khan Academy has excellent Linear Algebra Tutorials](https://www.khanacademy.org/#linear-algebra )

# Week 1

## Introduction

Machine learning is the science of getting computers to learn without being explicitly programmed

Supervised learning: Where the correct answers are given in training. Categorized into "regression" and "classification" problems.

* Regression problem: trying to map input variables to some continuous function.
- Classification problem: trying to predict results in a discrete output.

Unsupervised learning – deriving structure when we may not know the effects of the variables. Clustering based upon relationships.

## Linear Algebra

### Definitions
If $A$ is an $m \times n$ matrix ($m$ rows and $n$ columns), we say that "$A$ is an element of the set $\Bbb R ^ {m \times n}$". 

A *vector* is an $n \times 1$ matrix $\in \Bbb R ^n$, called an "$n$-dimensional vector", "column vector".

A vector can be 1-indexed or 0-indexed.  Assume 1-indexed for maths. 

### Matrices

### Multiplication
When multiplying a matrix by a column vector, the result will be a column vector with dimension of the number of rows in the matrix.

The "outer product" is a matrix of the dimensions of column vector $x$ row vector.
The "inner product" or "dot product" is a single real number (row vector $x$ column vector).

For a matrix multiplication $A \times B$ to be defined, the columns of $A$ need to equal the rows of $B$.

Let $A \in \Bbb R^{m \times n}$ and $B \in \Bbb R^{n \times p}$. Then:

$$\begin{align}
C & = AB \in \Bbb R^{m \times p} \\
C_{ij} & = \sum_{k=1}^n A_{ik}B_{kj}
\end{align}$$


When multiplying matrices $A \times B$

* Work across the columns of $B$, multiplying the row of $A$ by that vector.
* Work across the columns of $B$, taking the dot product with the row of $A$. Repeat, working down the rows of $A$. This builds the resultant top-to-bottom, then left-to-right. 

Video: https://www.youtube.com/watch?v=TwliA2BL_9g

#### Properties
* Not commutative: $A \times B \neq B \times A$ (except square matrix and identity matrix)
  The product may not even be defined if the matrices are not square.
* Associative: $(A \times B) \times C=A \times(B \times C)$

### Identity matrix
The identity matrix of dimensions $n \times n$ is denoted $I$ or $I_{n \times n}$ and defined by:
$I_{ij} = \begin{cases}
1, & \text{if}\; i=j \\
0, & \text{otherwise}
\end{cases}$

For any matrix $A$, $A \cdot I = I \cdot A = A$

### Inverse
The inverse of $A$ is denoted $A^{-1}$, and defined such that $A A^{-1} = A^{-1} A = I$

Only square matrices have an inverse.

If $A$ has determinant $|A|= 0$, then $A^{-1}$ does not exist. (The scalar 0 also does not have an inverse).

Matrices which don't have an inverse are called "singular" or "degenerate".

### Determinant
The determinant of $A$ is denoted $|A|$. Given
$\begin{align}A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}\end{align},\quad |A| = ad -bc$.

If we apply $A$ as the linear transformation of a unit square $U$ into $U_A$, then the determinant $|A|$ is the area of that transformed square. In a sense, the determinant is the size, or “norm”, of a square matrix.  The determinant of $I_2$ is $1$ since there is no reduction in size when used as a transformation.

### Transpose
The transpose of $A \in \Bbb R^{m \times n}$ is denoted $A^T \in \Bbb R^{n \times m}$ is given by $A^T_{ij} = A_{ji}$

Visually, the elements are reflected along the line of $1$s of the identity matrix.

### Practical uses
Multiplying a matrix of Cartesian $x,y$ coordinates by the matrix $\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}$ will reflect an object vertically.

A lot of computation can be performed by a matrix multiplication: e.g., testing 3 hypothesis at once. Here the multiplier's columns are $\theta$s and the resultant matrix's columns are $\hat y$.

![Data points multiplied by equations](http://imgur.com/ID6y2AHl.png)

## Linear Regression

Linear regression with one variable is also known as "univariate linear regression."

A tuple $(x^{(i)},y^{(i)})$ is called a training example, with $i$ being an index into the training set.

$x^{(i)}$ denotes an input example or *feature*, and $y^{(i)}$ denotes the output or *target* variable which we are trying to predict.
$\mathcal X$ denotes the space of input values, and $\mathcal Y$ the space of output values.  When the target variable is continuous, the problem is called a *regression* problem, otherwise it's a *classification* problem.

$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots$

The $\theta$s are the *parameters*, also called *weights* parameterising the space of linear functions mapping $\mathcal X$ to $\mathcal Y$.

The goal is: given a training set, to produce a function $h: \mathcal X \mapsto \mathcal Y$ where $h(x)$ is a "good" predictor of the corresponding value of $y$. For historical reasons, the function $h$ is called a *hypothesis*. It is the output of the learning algorithm, whose input is the training set.

To simplify notation, we let $x_0 = 1$ (called the *intercept term* (think $b$ in $y=mx+b$)). When $n$ is the number of input variables, and $\theta$ and $x$ are vectors:

$$h(x) = \sum_{i=0}^n \theta_i x_i = \theta^Tx$$

To learn the values of $\theta$, we need to make $h(x)$ as close as possible to $y$ for the training examples. For each value of $\theta$, the cost function measures how close $h(x^{(i)}))$ is to $y^{(i)}$.

### Mean Squared Error (Cost) Function

The most commonly used for linear regression problems.  *Ordinary least squares* regression given $m$ training examples:

$$ J(\theta) = \dfrac {1}{2m} \displaystyle \sum _{i=1}^m \left ( \hat{y}_{i}- y_{i} \right)^2 $$

Here, $\hat y$ represents the predicted equation for a line of best fit in linear regression. $\hat y_i$ is $h_\theta(x^{(i)}$).

Simplistically, this is $\frac 1 2 \bar x$ where $\bar x$ is the mean of the squares of $\hat y_i - y_i$

This function is also called the "Mean Squared Error". The $ \frac 1 2$ is cancelled out by the derivative of the squared term for convenience.

[Interactive, intuitive example](http://setosa.io/ev/ordinary-least-squares-regression/ )

### Derivative

* [Chris McCormicks detailed worked derivation (incl chain and power rule)](http://mccormickml.com/2014/03/04/gradient-descent-derivation/ )
* [Math.SE](https://math.stackexchange.com/a/1695446/389109 )
* [List of derivative rules](http://www.mathwords.com/d/derivative_rules.htm )
* [Wikipedia Differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules )

Notation: Partial derivative: $\partial$, derivative: $d$.

Derivative with only one feature:
$$\begin{align}\frac{\partial}{\partial\theta_0} J &= \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)})-y^{(i)}\right)\\
\frac{\partial}{\partial\theta_1} J &= \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)})-y^{(i)}\right) x^{(i)}\end{align}$$

Derivative with $n \ge 1$ features ($x_0^{(i)} = 1$):

$$\frac{\partial}{\partial\theta_j} J = \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)})-y^{(i)}\right) x_j^{(i)}$$

## (Batch) Gradient Descent

We want to find $\min_{\theta_0, \cdots, \theta_n} J(\theta_0, \cdots, \theta_n)$.

We want to choose the values in the vector $\theta$ so as to minimise $J(\theta)$. We start with an initial guess for $\theta$ (possibly all zeros) and repeatedly change it to minimise the cost.

Different starting points can lead to use ending up at a different local minimum.

The gradient descent algorithm repeatedly (and simutaneously) performs for all values of $j$:

$$ \theta_j := \theta_j - \alpha \frac \partial {\partial\theta_j} J(\theta) $$
Here $\alpha$ is called the learning rate, and determines how far $\theta_j$ moves in the in the opposite direction of the gradient. $\alpha$ remains fixed, with the step size changing based upon the gradient of $J$, taking smaller steps as the gradient of $J$ decreases with convergence.

If $\alpha$ is too small, convergence will require unnecessary iterations; if too large, it will fail to converge or even diverge (bouncing up and out of a simple parabola).

If already at a local minimum, $\theta_j$ will not change as the partial derivative $\frac {\partial }{\partial \theta_j} J(\theta) = 0$. For linear regression, the error squared cost function is a convex function, so there is only one global minimum or optimum.

For more than two dimensions, the analogue of the derivative is the gradient ($\nabla J$, pronounced "del J" in one video) which gives the vector of steepest descent. Its elements are partial derivatives of $J$ w.r.t. each dimension, or $\theta_j$.

All new values for vector $\theta$ are all calculated before being updated atomically.

Gradient descent works for any cost function, not just for linear regression.

*Batch Gradient Descent* uses the whole training set in each iteration. Some gradient descent only uses a subset of the training set.

Normal equations method (advanced linear algebra) can find the minimum of a function by a numerical non-iterative method, however linear gradient descent will scale better to larger data sets. 

TODO: [Watch YouTube vid with vector / matrix version](https://www.youtube.com/watch?v=WnqQrPNYz5Q ). Are vectors assumed to be row vectors here? Perhaps see prior.

## Other gradient descent

Stochastic gradient descent picks a point from the training set and updates $\theta$ only for that point, iterating over all training set data. May not work so well if the examples chosen are outliers.

Mini-batch has $ 1 < \text{batch size} < \text{size of training set}$

## Parametric and non-parametric algorithms

In a parametric model, we have a finite number of parameters, and in nonparametric models, the number of parameters is (potentially) infinite. In nonparametric models, the complexity of the model grows with the number of training data; in parametric models, we have a fixed number of parameters (or a fixed structure if you will).

> A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a parametric model. No matter how much data you throw at a parametric model, it won't change its mind about how many parameters it needs.
— [Artificial Intelligence: A Modern Approach](https://www.amazon.com/dp/0136042597?tag=inspiredalgor-20 ), page 737

Examples: linear regression, logistic regression, and linear Support Vector Machines; these have a fixed size of parameters (the weight coefficients.)

> Nonparametric methods are good when you have a lot of data and no prior knowledge, and when you don't want to worry too much about choosing just the right features.
— ibid

Examples: K-nearest neighbour, decision trees, RBF kernel SVMs; (the number of parameters grows with the size of the training set).

An RBF kernel SVM is non-parametric whereas a linear SVM is parametric because in the RBF kernel SVM, we construct the kernel matrix by computing the pair-wise distances between the training points.


[//]: # (This may be the most platform independent comment)
