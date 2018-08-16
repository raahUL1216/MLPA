function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

t = theta;
t(1,:) = 0;

h = X*theta;
J = 0.5/m * (h - y)' * (h - y) + (lambda*0.5/m)*t'*t;

grad = 1/m * X' * (h - y) + (lambda/m)*t;


%J = (sum((h-y) .^ 2) + (lambda * (sum(theta .^ 2)- theta(1,1) * theta(1,1))))/(2*m);


%grad(1, 1) = sum(h-y)/m;
%grad(1, 2) = (sum((h-y) .* X(:, 2)) + lambda * theta(2, 1))/m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
