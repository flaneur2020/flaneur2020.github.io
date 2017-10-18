function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


H = sigmoid(X * theta)
jSum = 0
for i=1:size(X)(1)
    jSum += 0 - y(i) * log(H(i)) - (1 - y(i)) * log(1-H(i))
end
J = jSum / m + theta' * theta * lambda / m / 2 - theta(1) * theta(1) * lambda / m / 2

gradT = lambda * theta
gradT(1) = 0
grad = X' * (sigmoid(X * theta) - y) / m + gradT / m







% =============================================================

end
