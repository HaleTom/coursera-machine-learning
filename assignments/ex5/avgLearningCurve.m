function [error_train, error_val] = ...
    avgLearningCurve(X, y, Xval, yval, samples=50, lambda=1)

% samples = number of random samples to take for each set size


% Implement ex5.pdf Section 3.5:
% Take n random samples of each set size and average the errors

%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

describe 3 samples lambda
pause

for set_size = 1:m
    printf (" === Set size: %d ===\n", set_size);
    error_train_total=0; % Accumulator to divide by samples
    error_val_total=0; % Accumulator to divide by samples
    for sample = 1:samples # Average over "samples" random samples
        selection = randperm (m, set_size);
        theta = trainLinearReg(X(selection,:), y(selection), lambda);
        error_train_total += linearRegCostFunction(X(selection,:), y(selection), theta, 0);
        error_val_total += linearRegCostFunction(Xval(selection,:), yval(selection), theta, 0);
    end
    error_train(set_size) = error_train_total / samples;
    error_val(set_size) = error_val_total / samples;
    % [error_val(set_size), ~] = linearRegCostFunction(Xval, yval, theta, 0);
end

% When you are computing the training set error, make sure you compute it on the
% training subset (i.e., X(1:n,:) and y(1:n)) (instead of the entire training
% set). However, for the cross validation error, you should compute it over the
% entire cross validation set.


% =========================================================================

end
