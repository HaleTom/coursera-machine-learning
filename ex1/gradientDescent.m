function [theta, J_history] = gradientDescent(X, y, theta=zeros(columns(X),1), alpha=0.001, num_iters=1e4)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

debug = false;
pre_tune_alpha = false;
use_epsilon = true;
epsilon=1e-9;
tune_alpha = false; % See FAIL below

if debug,
    printf('Called with alpha = %f, num_iters=%f\n', alpha, num_iters);
end

currentFig=gcf(); % Restore for assignment script at the end

% Tune alpha upwards
prev_cost = Inf;
cost = computeCost(X, y, theta);

alpha_scale=1.1; % Scaling factor
alphas=[]; % Store values for graphing
prev_cost = Inf;
cost = computeCost(X, y, theta);
try_alpha = alpha;
iter = 0;
while (pre_tune_alpha && cost < prev_cost),
    prev_cost = cost;

    alpha = try_alpha;
    alphas = [alphas try_alpha];

    try_alpha = alpha * alpha_scale;
    new_theta = update_theta(X, y, theta, try_alpha);
    cost = computeCost(X, y, new_theta);
    iter++;
end
if debug,
    % alpha = orig_alpha; iter=0;
    printf('Chose alpha = %f after %d iterations\n', alpha, iter);
end

if debug && pre_tune_alpha,
    % Plot alpha tuning
    alphaPlot=figure();
    plot(alphas);
    legend('alpha');
    title('Alpha tuning');
    xlabel('Iterations');
end

% Initialize some useful values
m = length(y); % number of training examples
J_history = [];

iter = 0;
cost = Inf;
prev_cost = -Inf;
while ++iter <= num_iters && abs(prev_cost - cost) > epsilon,
    prev_cost = cost;
    prev_theta = theta;

    theta = theta - alpha / m * X' * (X * theta - y);

    % Compute cost with new theta
    cost = computeCost(X, y, theta);
    if cost > prev_cost,
    % if cost > 10e10,
        printf('Bailing: strange cost in iteration %d\n', iter);
        theta = NaN;
        break;
    end

    % % Tune alpha after after prev_cost is setup
    % if tune_alpha && iter > 1, alpha *= .98 * prev_cost / cost; end
    % FAILS on: gradientDescent(X, y, [0;0], 0.015390, 1e5)

    % Save the cost J in every iteration
    J_history(iter) = cost;

    % Have we converged?
    if use_epsilon && abs(prev_cost - cost) < epsilon,
        break;
    end
end

if debug,
    iter -= 1 % Print steps to convergence
    cost
    % Plot cost against descent iterations
    figure();
    plot(J_history);
    title('J(theta)');
    xlabel('Iterations');
    figure(currentFig);
end

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
