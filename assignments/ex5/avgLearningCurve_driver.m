function avgLearningCurve_driver (X_poly, y, X_poly_val, yval, samples=50, lambda=1);

% Call after running ex5
% avgLearningCurve_driver (X_poly, y, X_poly_val, yval, 50, 0.01);

describe lambda
pause

m = length(y);

figure(1);
[error_train, error_val] = ...
    avgLearningCurve(X_poly, y, X_poly_val, yval, samples, lambda);
    plot(1:m, error_train, 1:m, error_val);

describe lambda
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

end
