# Machine Learning - Week 6 - Advice for Applying Machine Learning

When we have poor prediction performance, we can:

* Get more training examples (can be difficult, expensive)
* Try smaller sets of features (to prevent overfitting)
* Try additional features (which may be more informative)
* Try polynomial features
* Increase or decrease $\lambda$

Don't apply at random - gathering more training data could take months. For diagnosis and the decision guidance, see below.

## Machine learning diagnostics

Diagnostics can take some time to implement, but can be a very good use of time. Reconnaissance time is seldom wasted.


## How to evaluate a hypothesis

Due to overfitting, a hypothesis may have a low error for the training examples but fail to generalise to unseen examples. 

With many variables, visualising the hypothesis function to look for overfitting becomes difficult.

It's important to have a separate set of data (randomised, about 30% of total) to evaluate a hypothesis. 

The new procedure using these two sets is then:

1. Learn $\Theta$ and minimize $J_{train}(\Theta)$ using the training set
2. Compute the test set error $J_{test}(\Theta)$

### Test set error

The standard definitions of the cost functions can be used for $J_{test}(\Theta)$

Additionally, the *0/1 misclassifiction error* can be used:

$$err(h_\Theta(x),y) = \begin{cases} 1 & \mbox{if } h_\Theta(x) \geq 0.5\ and\ y = 0\ or\ h_\Theta(x) < 0.5\ and\ y = 1\newline 0 & \mbox otherwise \end{cases}$$

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

$$\text{Test Error} = \dfrac{1}{m_{test}} \sum^{m_{test}}_{i=1} err(h_\Theta(x^{(i)}_{test}), y^{(i)}_{test})$$

This gives us the proportion of the test data that was misclassified.


| Training | Test | Cause | 
|----------|------|-------|
| Low error| High error| Overfitting|



[//]: #speeling (check)
