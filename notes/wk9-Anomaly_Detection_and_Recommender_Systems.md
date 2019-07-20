## Anomaly Detection
Multivariate Gaussian anomaly detection requires <img src="/notes/tex/079bfec7814e7f4bcbae5a5e2830bb51.svg?invert_in_darkmode&sanitize=true" align=middle width=46.21760714999999pt height=17.723762100000005pt/> (for invertability of covarance matrix). Ng uses it when <img src="/notes/tex/c744d5fe5e9dc9218fa5f27cea17f65a.svg?invert_in_darkmode&sanitize=true" align=middle width=62.656025849999985pt height=21.18721440000001pt/>.

With many features this becomes computationally expensive, so consider PCA for feature reduction.

If using basic Gaussian anomaly detection, additional features are created to represent correlated values.

Covariance matrix will be non-invertible if there are features which are linearly dependant, eg, one is a duplicate feature or a feature is a weighted sum of two others.

## Recommender Systems

We don't include <img src="/notes/tex/08da0eff87a450c1af2ef3a27bf4243e.svg?invert_in_darkmode&sanitize=true" align=middle width=46.90628744999999pt height=21.18721440000001pt/> for the y intercept because we allow the model to learn whatever features it wants, including a feature which could be set to a constant if that aids the optimisation.
