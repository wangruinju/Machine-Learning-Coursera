function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.[]
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
trial = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
alls = zeros(numel(trial)* numel(trial), 3);
for i = 1:numel(trial)
    for j = 1:numel(trial)
        C_temp = trial(i);
        sigma_temp = trial(j);
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        alls((i-1)*numel(trial) + j,:) = [C_temp, sigma_temp, mean(double(predictions ~= yval))];
    end
end
[smallest_error, I] = min(alls(:,3));
C = alls(I,1);
sigma = alls(I,2);







% =========================================================================

end
