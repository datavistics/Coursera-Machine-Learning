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


h_th = sigmoid(X*theta);

% This is where the magic happens. I set the theta_0 term to 0 so that
% we dont have a lambda term for theta_0

scale = lambda * ones(length(theta), 1);;
scale(1) = 0;
l_term = dot(.5/m * scale .* theta, theta);

J = 1/m * (dot(-y , log(h_th)) - dot(( 1 - y) , log(1-h_th))) + l_term;


grad = 1/m * X' * (h_th - y) + 1/m * scale .* theta;


% =============================================================

end
