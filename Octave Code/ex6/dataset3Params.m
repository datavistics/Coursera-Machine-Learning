function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% This is not the best way, but its less frustration with octave
to_beat = 1000000

for C_test= [.01 .03 .1 .3 1 3 10 30]
  for sigma_test= [.01 .03 .1 .3 1 3 10 30]
    
    printf("C = %d; sigma = %d", C_test , sigma_test)
    model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test)); 
    predictions = svmPredict(model, Xval);
    result = mean(double(predictions ~= yval));
    
    if result < to_beat
      to_beat = result;
      C = C_test;
      sigma = sigma_test;
    endif
  endfor
endfor



% =========================================================================

end
