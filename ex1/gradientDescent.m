
function [theta, J_history] = gradientDescent(X, y, theta=zeros(columns(X),1), alpha=0.001, num_iters=50000)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

currentFig=gcf(); % Restore for assignment script at the end

% Initialize some useful values
m = length(y); % number of training examples
J_history = [];

epsilon=10^-5;
iter = 0;

% Tune alpha upwards
alpha_scale=1.9 ; % Scaling factor
alphas=[alpha]; % Store values for graphing

prev_cost = Inf;
cost = computeCost(X, y, theta);
cost
prev_cost

% Tune alpha upwards
alpha_scale=1.2 ; % Scaling factor
alphas=[]; % Store values for graphing
 
prev_cost = Inf;
cost = computeCost(X, y, theta);
 
try_alpha = alpha;
new_theta = theta;
while (cost < prev_cost),
    % prev_cost = computeCost(X, y, theta);
    % if cost < prev_cost,
    prev_cost = cost;
    % theta = new_theta;
    % J_history(++iter) = cost;
 
    % printf('Alpha: %8.3f, Cost: %8.3f\n', alpha, cost);
 
    alpha = try_alpha;
    alphas = [alphas try_alpha];
 
    try_alpha = alpha * alpha_scale;
    new_theta = update_theta(X, y, theta, try_alpha);
    cost = computeCost(X, y, new_theta);
    iter++;
end

% alphas = alphas'
iter
theta

% Plot alpha tuning
alphaPlot=figure();
plot(alphas);
legend('alpha');
title('Alpha tuning');
xlabel('Iterations');

% alpha=0.001; iter=0;
printf('Chose alpha = %f after %d iterations\n', alpha, iter);

iter = 0;
cost = Inf;
prev_cost = -Inf;

first_loop = true;
while iter++ <= num_iters && abs(prev_cost - cost) > epsilon,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    prev_cost = cost;
    prev_theta = theta;

    theta = theta - alpha / m * X' * (X * theta - y);

    % Compute cost with new theta
    cost = computeCost(X, y, theta);
    % if cost > prev_cost,
    if cost > 10e10
        printf('Bailing: cost out of bounds in iteration %d\n', iter);
        theta = NaN;
        break;
    end

    % Tune alpha after some time for convergence
    if !first_loop, alpha *= prev_cost / cost; end
    % alpha *= prev_cost / cost;

    % Save the cost J in every iteration
    J_history(iter) = cost;

    % Have we converged?
    if abs(prev_cost - cost) < epsilon,
        break;
    end
    first_loop = false; % Not in the first loop anymore
    % ============================================================
end

iter
% Plot cost against descent iterations
figure();
plot(J_history);
title('J(theta)');
xlabel('Iterations');
figure(currentFig);

end

function newtheta = update_theta (X, y, theta, alpha)
    % newtheta = theta - alpha / m * X' * (X * theta - y);
    newtheta = theta - alpha / length(y) * X' * (X * theta - y);
end

function debug_point()
    d = dbstack(1);
    d = d(1);
    fprintf('Reached file %s, function="%s", line %i\n', d.file, d.name, d.line)
end
