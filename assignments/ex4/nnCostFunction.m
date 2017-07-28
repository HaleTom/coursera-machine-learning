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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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
hX = predict(Theta1, Theta2, X);
J = 1/m * sum(sum((-y .* log(hX) - (1 - y) .* log(1 - hX))));

% Add regularisation
for i = 1:length(Theta)
  Theta{i}(:,1) = zeros(rows(Theta{i}), 1); % Zero bias terms
  J += lambda / 2 / m * sum(sum(Theta{i} .* Theta{i}));
end

%
% Compute gradients via backpropagation
%

% Inintialise Thetas via Normalised Initialisation
% https://stats.stackexchange.com/questions/291777/why-is-sqrt6-used-to-calculate-epsilon-for-random-initialisation-of-neural-net
for i = 1:length(Theta)
  Theta{i} = randInitializeWeights(columns(Theta{i})-1, rows(Theta{i}));
end



%----------------------

[Theta1, Theta2] = Theta{1:2};
Theta1 = Theta{1};
Theta2 = Theta{2};

yy = y;
oldX = X;
X = [ones(m,1) X];
for t=1:m
  % forward pass
  a1 = X(t,:);
  z2 = Theta1*a1';
  a2 = [1; sigmoid(z2)];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);

  % backprop
  delta3 = a3-yy(t,:)';
  delta2 = (Theta2'*delta3).*[1; sigmoidGradient(z2)];
  delta2 = delta2(2:end);

% DEBUG
  if t == 1500
    delta2_test = delta2;
    % describe delta2_1
  end

  Theta1_grad = Theta1_grad + delta2*a1;
  % describe Theta1_grad delta2 a1;
  Theta2_grad = Theta2_grad + delta3*a2';
  % describe Theta2_grad delta3 a2;
end

Theta1_grad = (1/m)*Theta1_grad+(lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad+(lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
X = oldX;

% return

% ------------------------

% Get Z (non-activated output) of all neurons in network
[hX, Z] = predict(Theta1, Theta2, X);

% Get error of output layer
layers = 1 + length(Theta);
d{layers} = hX - y;

%   % backprop
%   delta3 = a3-yy(t,:)';
%   delta2 = (Theta2'*delta3).*[1; sigmoidGradient(z2)];
%   delta2 = delta2(2:end);

% Propagate errors backwards through hidden layers
for layer = layers-1 : -1 : 2
  d{layer} = d{layer+1} * Theta{layer};
  d{layer} = d{layer}(:, 2:end); % Remove "error" for constant bias term
  d{layer} .*= sigmoidGradient(Z{layer});
  % d{layer} .*= sigmoidGradient([ones(m,1) Z{layer}]);
  % describe layer Theta{layer} d{layer+1} sigmoid(Z{layer}) d{layer};
end

% delta2s are EQUAL.
% disp( delta2_test - d{2}(1500,:)')
% describe delta2_1
% disp(delta2_1)
% describe d{2}(1,:)
% disp(d{2}(1,:)')
% ----- end delta2 EQUAL --------

% "Theta1_grad" is a matrix of size [25 401]
% "delta2" is a matrix of size [25 1]
% "a1" is a matrix of size [1 401]
%
% "Theta2_grad" is a matrix of size [10 26]
% "delta3" is a matrix of size [10 1]
% "a2" is a matrix of size [26 1]%

% "layer" is a double = 2
% "Theta{layer}" is a matrix of size [10 26]
% "d{layer+1}" is a matrix of size [5000 10]
% "sigmoid(Z{layer})" is a matrix of size [5000 25]
%
% "layer" is a double = 1
% "Delta{layer}" is a matrix of size [25 401]
% "d{layer+1}" is a matrix of size [5000 25]
% "sigmoid(Z{layer})" is a matrix of size [5000 400]

%%%%%%%%
  Theta1_grad = Theta1_grad + delta2*a1;
  Theta2_grad = Theta2_grad + delta3*a2';
%%%%%%%%


% Zero Theta bias weights for later regularisation
for layer = 1:length(Theta)
  Theta{layer}(:,1) = ones(rows(Theta{layer}), 1);
end

% Calculate Theta gradients
for layer = 1:layers-1
  % Create empty grad_Theta as accumulator
  grad_Theta{layer} = zeros(size(Theta{layer}));
  % describe z{layer}
  for i = 1:m
    % grad_Theta{layer} += d{layer+1}(i,:)' * sigmoid([1 Z{layer}(i,:)]); % Outer product
    grad_Theta{layer} += d{layer+1}(i,:)' * ([1 Z{layer}(i,:)]); % Outer product
  end
  % describe Z{layer}(m,:);
  % disp ([1 Z{layer}(m,:)]);

  grad_Theta{layer} = 1/m * (grad_Theta{layer});
  % grad_Theta{layer} = 1/m * (grad_Theta{layer} + lambda * Theta{layer});

  describe layer grad_Theta{layer} d{layer+1} sigmoid(Z{layer})
  % describe grad_Theta{layer}
end

% Unroll gradients
grad=[];
for i = 1:length(grad_Theta)
  % printf("size(grad_Theta{i})")
  % size(grad_Theta{i})
  grad = [grad; grad_Theta{i}(:)];
  % describe grad
  % describe grad_Theta{i}
end

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
