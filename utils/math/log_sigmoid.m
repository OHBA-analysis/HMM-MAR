function z = log_sigmoid(x)
% computes and returns the logistic sigmoid of x. Operation is performed 
% element-wise if x is a vector or matrix.
%
%       z = 1 ./ (1-exp(-x))

z = (1+exp(-x)).^-1;

end