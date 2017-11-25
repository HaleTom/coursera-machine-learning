% ex5.pdf, section 3.4:
% Calculate the error of the test set using the best found lambda (3)
function testError(Xtrain, ytrain, Xval, yval, Xtest, ytest, lambda)

% Compute test set error at lowest error (lambda==3)
theta = trainLinearReg(Xtrain, ytrain, lambda);

J = linearRegCostFunction(Xval, yval, theta, 0);
printf('Val  set error with lambda==%f is: %f\n', lambda, J);

J = linearRegCostFunction(Xtest, ytest, theta, 0);
printf('Test set error with lambda==%f is: %f\n', lambda, J);

end

