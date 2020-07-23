

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% H = zeros(size(X)(1))

H = sigmoid(X * theta)
jSum = 0
for i=1:size(X)(1)
    jSum += 0 - y(i) * log(H(i)) - (1 - y(i)) * log(1-H(i))
end
J = jSum / m

grad = X' * (sigmoid(X * theta) - y) / m


% =============================================================

end


function h = hFunction(theta, x)
    z = x * theta
    h = 1 / (1 + e ^ (0 - z))
end
