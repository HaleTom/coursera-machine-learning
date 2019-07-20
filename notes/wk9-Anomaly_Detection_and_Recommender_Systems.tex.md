## Anomaly Detection
Multivariate Gaussian anomaly detection requires $m > n$ (for invertability of covarance matrix). Ng uses it when $m > 10n$.

With many features this becomes computationally expensive, so consider PCA for feature reduction.

If using basic Gaussian anomaly detection, additional features are created to represent correlated values.

Covariance matrix will be non-invertible if there are features which are linearly dependant, eg, one is a duplicate feature or a feature is a weighted sum of two others.

## Recommender Systems

We don't include $x_0 = 1$ for the y intercept because we allow the model to learn whatever features it wants, including a feature which could be set to a constant if that aids the optimisation.
