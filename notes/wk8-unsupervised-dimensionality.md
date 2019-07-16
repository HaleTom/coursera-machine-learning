## K-means

If there is a K-means class with zero elements assigned to it, then either remove that class or randomly reinitialise it somewhere else.

With lower number (2-10) of K classes, random initialisation will give a larger improvement in results. With higher K, the first random pick will probably be quite good.

## PCA

Pre-req:

* Mean normalisation, and
* Feature scaling

With dimensionality reduction, it's up to a human to determine what the new features represent.

Projection error - distance of each point to the projection surface. The vectors chosen minimise this error.

Whether the PCA output projection vectors are positive or negative is irrelevant - the still define the same line or (hyper)plane

We then project the data onto the linear subspace space spanned by the PCA vectors.

With linear regression, the vertical distance is minimised w.r.t a special variable (often <img src="/notes/tex/deceeaf6940a8c7a5a02373728002b0f.svg?invert_in_darkmode&sanitize=true" align=middle width=8.649225749999989pt height=14.15524440000002pt/>), with PCA the orthogonal distances are minimised.

The proof that the PCA algorithm works is extremely complicated.

SVD = Singular Value Decomposition

Matlab `svd` is more numerically stable than using `eig`.

Covariance matrix is symmetric positive semi-definite, so in this case using `eig` would be ok too.

Covariance matrix will be n x n size.

The columns of the U matrix (also n x n) returned by `svd` will be the u vectors we want. We just take the first <img src="/notes/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> column vectors.
