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

A1 = X;
[A2, A3, Z2, Z3] = feedForward(Theta1, Theta2, X);
J = calculateJ(A2, A3, X, y, lambda);

rSum = sum(Theta1(:, 2:end)(:) .^ 2) + sum(Theta2(:, 2:end)(:) .^ 2);
J = J + rSum * lambda / (2 * m);

%
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

Y = zeros(m, num_labels);
for i = 1:num_labels
    Y(:, i) = (y == i);
end

disp('size(X);'); disp(size(X));  % 500 x 400
disp('size(A2);'); disp(size(A2));  % 5000 x 25
disp('size(A3);'); disp(size(A3));  % 5000 x 10
disp('size(Y);'); disp(size(Y));  % 5000 x 10
disp('size(Theta2);'); disp(size(Theta2));  % 10x26
disp('size(Z2);'); disp(size(Z2));  % 5000 x 25



delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));
disp('size(delta_2);'); disp(size(delta_2));  % 5000 x 25
for i = 1:m
    e3 = A3(i, :) - Y(i, :); % 1 x 10
    e2 = (e3 * Theta2)(:, 2:end) .* sigmoidGradient(Z2(i, :)); % 1 x 25
    delta_2 = delta_2 + [e3' * [1 A2(i, :)]]; % 10 x 26
    delta_1 = delta_1 + [e2' * [1 A1(i, :)]];
end

% disp('delta_2'); disp(delta_2);

Theta1_grad = delta_1 / m;
Theta2_grad = delta_2 / m;

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

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function J = calculateJ(A2, A3, X, y, lambda)
m = size(X, 1);
num_labels = size(A3, 2);
% calculate the J value
% [A1, A2] = feedForward(Theta1, Theta2, X)
H = A3;

jSum = 0;
for k = 1:num_labels
    Hk = H(:, k);
    Yk = (y == k);
    jSum += Yk' * log(Hk) + (1 - Yk)' * log(1 - Hk);
    % for i = 1:m
    %     jSum += Yk(i) * log(Hk(i)) + (1 - Yk)(i) * log(1 - Hk(i))
    %end
end
% need to strip the bias vectors, which are Theta1(:, 1) and Theta2(:, 1)
J = 0 - jSum / m
end


function [A2, A3, Z2, Z3] = feedForward(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

Z2 = [ones(m, 1) X] * Theta1';
A2 = sigmoid(Z2);
Z3 = [ones(m, 1) A2] * Theta2';
A3 = sigmoid(Z3);
end
