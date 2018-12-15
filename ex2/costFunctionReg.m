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
% J  = 1/m (- y * log(h(x)) - ((1 - y) * log(1-h(x)))) + lambda /2m .* theta^2
J = -y' * log(sigmoid(X * theta));
J = (J - ((1-y)' * log(1-sigmoid(X * theta)))) .* ( 1/m);
reg = (lambda/(2*m))*(sum(theta(2:length(theta),:) .^ 2));
J = J + reg;
%h = sigmoid(X*theta);
%reg = (lambda/(2*m))*(sum(theta(2:length(theta),:) .^ 2));
%J = (1/m)*((-y' * log(h)) - ((1-y)' * log(1-h))) + reg;

theta1 = theta(1, :);
%J = J - (((1. - y)' * log(1 - sigmoid(X * theta))).*1/m);
%J = J -  ((lambda/2*m) .* theta.^2);
%X(:,1) = [];

%theta1 = 1/m .* (sigmoid((X * theta) - y)' * X(:,1));

%theta1 = theta1 - (((1.-y)' * log(1-sigmoid(X * theta)))) .*1/m;
%theta(1,:) = [];
%X(:,1) = Xc;
%X(:,1) = [];
reg = (lambda/m) * theta;
reg(1,:) = 0;
grad = (X'*(sigmoid(X*theta) - y)) ./ m;
grad = grad + reg;

%theta = (1/m .* (sigmoid(X * theta) - y)' * X) + (lambda/m) * theta ;
%theta(1,:) = theta1;
%grad = theta;
%theta = -y' * log(sigmoid(X * theta));
%theta = theta - (((1. - y)' * log(1 - sigmoid(X * theta))).*1/m);
%theta = theta + ((lambda/2*m) .* theta.^2);
%J = theta1 + theta;
%mymatrix([2,4],:) = []
%h = sigmoid(X*theta);
%reg = (lambda/(2*m))*(sum(theta(2:length(theta),:) .^ 2));
%J = (1/m)*((-y' * log(h)) - ((1-y)' * log(1-h))) + reg;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
