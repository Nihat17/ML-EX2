function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%J =  1/m * (y * log(sigmoid(x))+(1-y) * log(1-sigmoid(x)))
% Initialize some useful values
m = length(y); % number of training examples
%h(x) = 1/1-e^-theta' *X
%theta is vector
% You need to return the following variables correctly 
J = 0;
%disp(sprintf('X : %0.1f',size(X)));
%disp(sprintf('X : %0.3f',X));
J  =   -y' * log(sigmoid(X * theta)); % * log(1-sigmoid(X * theta)));
J = (J - ((1.-y)' * log(1-sigmoid(X * theta)))) .*1/m;
%disp(sprintf(' J : %0.5f',J));
alpha = 0.001 * 3;
grad = zeros(size(theta));
grad = ((sigmoid(X * theta) - y)' * X)  .*1/m;
%grad = theta - alpha * sigma

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
