function [p, activation] = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted output layer activation
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


allTheta{1} = Theta1;
allTheta{2} = Theta2;

% size(X) == 5000 * 401 -- 5000 examples, 400 features
% size(Theta1) == 25 x 401  -- 25 nodes in hidden layer
% size(A1) == 5000 x 26  -- 25 hidden nodes + 1 bias
% size(Theta2) == 10 x 26 -- 10 class output

activation{1} = X; % Input layer

for Theta = 1:size(allTheta,2) % Cell array layout is like row vector
% Add bias node
	bias_added = [ones(rows(activation{Theta}), 1) activation{Theta}];

	activation{Theta + 1} = sigmoid(bias_added * allTheta{Theta}');
end

p = activation{end};

% =========================================================================

end
