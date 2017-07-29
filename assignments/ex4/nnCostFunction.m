function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Allow arbitrary network architectures. Create cell array of all Theta parameters
Theta={Theta1; Theta2};

% Transform y from integers in 1:10 into vectors which would be returned by the
% output layer
yOutput = zeros(length(y), rows(Theta{end}));

K = rows(Theta{end}); % Number of classes

for i = 1:length(y)
  % Expect integers from 1:
  if (!(class = y(i)) == floor(class) || class < 1 || class > K) 
    printf("unexpected value y(%d) = %f. Bailing.\n", i, class);
    return
  end
  yOutput(i, y(i)) = 1;
  % Or, faster (but no validation):
  % yv=[1:num_labels] == y % Use Broadcasting
  % Or
  % yv = bsxfun(@eq, y, 1:num_labels);
end
y = yOutput;

%
% Compute unregularised cost (J)
%

% Get h(X) and z (non-activated output of all neurons in network)
[hX, z, activation] = predict(Theta1, Theta2, X);

hX = predict(Theta1, Theta2, X);
J = 1/m * sum(sum((-y .* log(hX) - (1 - y) .* log(1 - hX))));

% Add regularisation
for i = 1:length(Theta)
  J += lambda / 2 / m * sum(sum(Theta{i}(:,2:end) .^ 2));
end

%
% Compute gradients via backpropagation
%

% Get error of output layer
layers = 1 + length(Theta);
d{layers} = hX - y;

% Propagate errors backwards through hidden layers
for layer = layers-1 : -1 : 2
  d{layer} = d{layer+1} * Theta{layer};
  d{layer} = d{layer}(:, 2:end); % Remove "error" for constant bias term
  d{layer} .*= sigmoidGradient(z{layer});
end

% Calculate Theta gradients
for l = 1:layers-1
  Theta_grad{l} = zeros(size(Theta{l}));

  % Sum of outer products
  Theta_grad{l} += d{l+1}' * [ones(m,1) activation{l}];

  % Add regularisation term
  Theta_grad{l}(2:end) += lambda * Theta{l}(2:end);
  Theta_grad{l} /= m;
end

% Unroll gradients
grad=[];
for i = 1:length(Theta_grad)
  grad = [grad; Theta_grad{i}(:)];
end

% ------- End of Ravi's code --------

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

end
